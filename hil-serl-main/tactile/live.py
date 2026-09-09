"""Live tactile heatmaps off capture.py's shared memory.

The control loop cannot afford to compute optical flow inline. franka_env.py:229
sleeps to hit 10 Hz *before* _get_obs() runs, so everything the observation costs
is added on top of the period: 31 ms of Farneback would push 100 ms to 131 ms,
and pi0.5 inference lands on top of that. A background thread keeps the flow off
the control path — the same shape as tactile/viewer.py's Pipeline, with the JPEG
stream replaced by "keep the latest heatmap".

    src = LiveTactileSource()          # blocks until the first heatmap exists
    heat = src()                       # (240, 320, 3) uint8 RGB, no work done here
    src.recalibrate()                  # when the gel is known to be untouched
    src.close()

Requires tactile/capture.py to already be running. It must also not be restarted
underneath us: a restart creates a fresh shared memory segment while this reader
stays mapped to the old one, and the frames simply stop advancing.
"""
import json
import threading
import time
from pathlib import Path

import cv2
from tactile.flow import TactileFlow, _colorize
from tactile.shmframe import FrameReader, read_distinct

# What viewer.py and record.py actually use. TactileFlow's own default is
# (960, 540), which measures 74 ms against 31 ms for no accuracy gain — known
# translations still recover to under 0.01 px (tactile/README.md perf table).
FLOW_SIZE = (640, 360)
# OpenCV takes (width, height); this yields the (240, 320, 3) the tactile
# observation key is declared with.
OUT_SIZE = (320, 240)
# Fixed colour scale. Per-frame normalisation would make heatmaps from different
# moments incomparable, so the same contact would look different across episodes.
VMAX = 3.0


class TactileStale(RuntimeError):
    """No new frame for too long — capture.py died or was restarted."""


class LiveTactileSource:
    """Callable returning the most recent tactile heatmap.

    Args:
        shm_name: shared memory segment, matching capture.py's --shm.
        ref_frames: frames averaged into the no-contact baseline.
        stale_s: raise if no new camera frame arrives for this long.
        save_dir: if set, the baseline is written here for offline replay.
    """

    def __init__(self, shm_name="tactile_frames", ref_frames=30, stale_s=2.0,
                 hz=15.0, save_dir=None):
        self.stale_s = stale_s
        self.ref_frames = ref_frames
        # Throttled like viewer.py:66-72. The camera runs at 30 Hz and the flow
        # costs ~31 ms, so an unthrottled loop burns most of a core to produce
        # frames the 10 Hz control loop then throws away.
        self.interval = 1.0 / hz if hz > 0 else 0.0
        try:
            self.rd = FrameReader(shm_name)
        except FileNotFoundError as e:
            # The bare errno here names the segment and nothing else, and this
            # is the failure everyone hits first.
            raise TactileStale(
                f"no shared memory segment {shm_name!r}. Start the camera first:"
                f"\n    python tactile/capture.py --shm {shm_name}"
            ) from e
        # ema=1.0 disables flow.py's cross-frame IIR. At the default 0.5 the
        # observation would depend on how recently the thread last ran, and a
        # reset would leak the previous episode's deformation into the first
        # frames of the next one — neither is visible in the observation itself.
        self.tf = TactileFlow(size=FLOW_SIZE, ema=1.0)

        self._lock = threading.Lock()
        self._heat = None
        self._err = None
        self._want_recal = threading.Event()
        self._stop = threading.Event()

        self._baseline()
        if save_dir is not None:
            self._save_baseline(Path(save_dir))

        # Produce the first heatmap before returning: TactileVLAWrapper.__init__
        # calls this immediately to size the observation space.
        got = self._await_frame()
        self._heat = self._to_heatmap(got[0])

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------------

    def __call__(self):
        """Latest heatmap, (240, 320, 3) uint8 RGB. Raises if the thread died."""
        with self._lock:
            if self._err is not None:
                # A dead daemon thread would otherwise just freeze the heatmap,
                # and the policy would keep training on a stale image with
                # nothing anywhere reporting a problem.
                raise self._err
            return self._heat

    def recalibrate(self):
        """Rebuild the baseline. The gel must be untouched when this runs.

        The gel creeps and the exposure drifts, so a baseline taken at startup
        degrades over a long session (tactile/README.md). Because _colorize uses
        a fixed vmax, that drift eats the dynamic range rather than showing up
        as an offset.
        """
        self._want_recal.set()

    def close(self):
        self._stop.set()
        # Long enough to cover a recalibration in flight: read_distinct blocks
        # up to 15 s, and closing the reader out from under it would free the
        # arrays it is copying from.
        self._thread.join(timeout=20.0)
        if self._thread.is_alive():
            print("[tactile] flow thread did not stop; leaving shm mapped")
            return
        self.rd.close()

    # -- internals ------------------------------------------------------------

    def _baseline(self):
        try:
            frames = read_distinct(self.rd, self.ref_frames)
        except SystemExit as e:
            # read_distinct raises SystemExit (shmframe.py:117), which slips
            # past every `except Exception` and exits with no traceback.
            raise TactileStale(
                f"capture.py holds {self.rd.shm.name!r} but has published no "
                "frames — check /dev/video0 and its log"
            ) from e
        # read_distinct returns whatever it collected when it times out, so a
        # partial baseline would silently be averaged from a single frame.
        if len(frames) < self.ref_frames:
            raise TactileStale(
                f"only {len(frames)}/{self.ref_frames} baseline frames; "
                "is tactile/capture.py running?"
            )
        self.tf.set_reference(frames)

    def _save_baseline(self, d):
        d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / "tactile_reference.png"), self.tf.ref_bgr)
        (d / "tactile_params.json").write_text(
            json.dumps({"flow_size": list(FLOW_SIZE), "out_size": list(OUT_SIZE),
                        "vmax": VMAX, "ref_frames": self.ref_frames,
                        "ema": 1.0}, indent=2),
            encoding="utf-8",
        )

    def _await_frame(self, last_seq=-1):
        deadline = time.monotonic() + self.stale_s
        while time.monotonic() < deadline:
            got = self.rd.latest()
            if got is not None and got[1] != last_seq:
                return got
            time.sleep(0.002)
        raise TactileStale(
            f"no new tactile frame for {self.stale_s}s — capture.py stopped, or "
            "it restarted and this reader is mapped to the old segment"
        )

    def _to_heatmap(self, frame):
        mag = self.tf.process(frame)["mag"]
        heat = _colorize(mag, VMAX, cv2.COLORMAP_TURBO)
        # applyColorMap returns BGR, but get_im() hands the cameras over as RGB
        # (franka_env.py:266) and both feed the same frozen ImageNet trunk,
        # whose per-channel normalisation is not symmetric.
        heat = heat[..., ::-1]
        return cv2.resize(heat, OUT_SIZE, interpolation=cv2.INTER_AREA)

    def _run(self):
        last = -1
        next_at = time.monotonic()
        try:
            while not self._stop.is_set():
                if self.interval:
                    next_at += self.interval
                    delay = next_at - time.monotonic()
                    if delay > 0:
                        self._stop.wait(delay)
                    elif delay < -1.0:
                        next_at = time.monotonic()  # fell too far behind
                    if self._stop.is_set():
                        break
                if self._want_recal.is_set():
                    self._want_recal.clear()
                    self._baseline()
                got = self._await_frame(last)
                last = got[1]
                heat = self._to_heatmap(got[0])
                with self._lock:
                    self._heat = heat
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            with self._lock:
                self._err = e
