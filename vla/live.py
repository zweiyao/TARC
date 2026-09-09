"""π0.5 action chunks at a fraction of the control rate, off the control loop.

franka_env.py:231 sleeps to hit 10 Hz *before* _update_currpos() and _get_obs()
run, so everything the observation costs is added on top of the period rather
than absorbed by it. A synchronous pi0.5 call in TactileVLAWrapper therefore
costs frequency one-for-one: measured on this box, inference is 81 ms (std 1.9,
max 85 over 20 calls), which drops the loop from ~9.3 Hz to ~5.3 Hz while
ACTION_SCALE and the compliance parameters are tuned for 10.

Calling pi0.5 every 5th step synchronously would not fix it — four steps at
107 ms followed by one at 188 ms. ACTION_SCALE turns the policy's output into a
*per-step* Cartesian delta, so an uneven period is an uneven commanded velocity.
Uneven is worse than uniformly slow.

So inference moves to a background thread and the control loop reads whatever
chunk is currently published, exactly like tactile/live.py. The difference is
the trigger: this one fires on a *step count* rather than a wall clock, which
keeps the ratio to the policy exactly 5:1 no matter how the loop jitters.

    src = LiveActionChunkSource()      # blocks until the first chunk exists
    chunk = src(obs)                   # (T, 8) float32, no inference here
    chunk = src.refresh(obs)           # blocking; for episode boundaries
    src.close()

Requires the pi0.5 service to already be running — see vla/README.md.
"""
import threading
import time

import numpy as np

from vla import pi05_client

# Assembled in this order to make the 19-dim state; must match the probe below
# and STATE_DIM in experiments/tarc/config.py.
STATE_KEYS = ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")

# Matches the resized camera observation. pi05_client.to_request pads everything
# to 224x224 anyway, so this only has to be shaped like a real frame.
PROBE_IMAGE = (1, 128, 128, 3)


class VLAStale(RuntimeError):
    """A request was sent and never came back — the pi0.5 server is wedged."""


class LiveActionChunkSource:
    """Callable returning the most recent π0.5 action chunk.

    Args:
        every: steps between inference requests. 5 gives 2 Hz against a 10 Hz
            control loop.
        chunk_t: action horizon the served checkpoint must return. The learner
            never connects, so its observation space has to assume one; a
            mismatch is caught here rather than as a broadcast error on the
            learner's first insert.
        state_dim: width of the assembled state vector, checked every request.
        stale_s: raise if a sent request has gone unanswered this long.
        client: injected for testing; defaults to a real connection.
    """

    def __init__(self, every=5, chunk_t=15, state_dim=19, stale_s=5.0,
                 client=None):
        self.every = every
        self.chunk_t = chunk_t
        self.state_dim = state_dim
        self.stale_s = stale_s
        self.client = pi05_client.connect() if client is None else client

        self._lock = threading.Lock()
        self._chunk = None
        self._err = None
        self._pending = None      # request handed to the thread, not yet taken
        self._busy = False
        self._sent_at = 0.0
        self._n = 0               # steps since the last request was issued
        # Bumped by refresh() to invalidate an in-flight inference: without it
        # a request issued before a reset would land afterwards and overwrite
        # the fresh chunk with a plan for the previous episode's arm pose.
        self._gen = 0

        self._wake = threading.Event()
        self._stop = threading.Event()

        # A valid chunk has to exist before __init__ returns: TactileVLAWrapper
        # (wrapper.py:161) calls this immediately to size the observation space.
        # This first call also compiles — measured at 14 s against 81 ms steady
        # state — which is indistinguishable from a hang with nothing on stdout.
        print("[vla] warming up pi0.5 (first call compiles, ~15s)...", flush=True)
        t0 = time.monotonic()
        self._chunk = self._infer(self._probe_request())
        print(f"[vla] ready in {time.monotonic() - t0:.1f}s, "
              f"chunk {self._chunk.shape}", flush=True)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------------

    def __call__(self, obs):
        """Latest chunk, (T, 8) float32. Issues a request every `every` calls.

        `obs` is the pre-SERLObsWrapper dict, so images are under obs["images"]
        and the state is still split across keys. obs=None is the space probe.
        """
        with self._lock:
            if self._err is not None:
                # A dead daemon thread would otherwise freeze the chunk, and the
                # policy would keep training against a stale plan with nothing
                # anywhere reporting a problem.
                raise self._err
            if obs is None:
                return self._chunk
            if self._busy and time.monotonic() - self._sent_at > self.stale_s:
                raise VLAStale(
                    f"pi0.5 has not answered for {self.stale_s}s. Check the "
                    f"service:\n    bash vla/serve_pi05.sh"
                )
            # `not self._busy` is the backpressure: if inference runs longer
            # than `every` steps the cadence degrades to the next multiple of
            # `every` rather than queueing requests, which would let the lag
            # grow without bound.
            if self._n % self.every == 0 and not self._busy:
                self._pending = self._request(obs)
                self._busy = True
                self._sent_at = time.monotonic()
                self._wake.set()
            self._n += 1
            return self._chunk

    def refresh(self, obs):
        """Infer now, blocking, and publish the result.

        For episode boundaries. Without it the first couple of steps of every
        episode carry the chunk computed at the *end* of the previous one, when
        the arm was somewhere else entirely. reset() already spends seconds in
        interpolate_move, so one inference is free there.
        """
        with self._lock:
            self._gen += 1            # discard whatever is in flight
            gen = self._gen
            # Only clear _busy for a request the thread has not picked up yet:
            # that one is genuinely cancelled here and the thread will never
            # reach the code that clears the flag, so leaving it set would wedge
            # the cadence permanently. A request already being inferred keeps
            # _busy set until the thread returns — it really is still in flight,
            # and clearing it early would let a second one be issued alongside.
            if self._pending is not None:
                self._pending = None
                self._busy = False
        chunk = self._infer(self._request(obs))
        with self._lock:
            if self._gen == gen:      # no further refresh happened meanwhile
                self._chunk = chunk
                # 1, not 0: this call already produced the chunk for the reset
                # observation, so the cadence resumes as though this had been
                # the firing step and the next request is `every` steps out.
                self._n = 1
        return chunk

    def close(self):
        self._stop.set()
        self._wake.set()
        # Long enough to outlast an inference in flight.
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            print("[vla] inference thread did not stop")

    # -- internals ------------------------------------------------------------

    def _probe_request(self):
        return {
            "wrist_1": np.zeros(PROBE_IMAGE, np.uint8),
            "wrist_2": np.zeros(PROBE_IMAGE, np.uint8),
            "state": np.zeros((1, self.state_dim), np.float32),
        }

    def _request(self, obs):
        state = np.concatenate(
            [np.atleast_1d(obs["state"][k]).ravel() for k in STATE_KEYS]
        )[None].astype(np.float32)
        # If this ever disagrees with the probe, the request silently carries
        # different numbers than the observation space claims and pi05_client's
        # slicing lands on the wrong fields.
        assert state.shape[1] == self.state_dim, (state.shape, self.state_dim)
        # No defensive copy: _get_obs() returns copy.deepcopy (franka_env.py:483)
        # so these arrays are private to this observation, and _update_currpos
        # rebinds rather than writing in place (:445).
        return {
            "wrist_1": obs["images"]["wrist_1"][None],
            "wrist_2": obs["images"]["wrist_2"][None],
            "state": state,
        }

    def _infer(self, req):
        chunk = pi05_client.infer(self.client, req).astype(np.float32)
        if chunk.shape[0] != self.chunk_t:
            raise ValueError(
                f"pi0.5 returned T={chunk.shape[0]}, but the learner's "
                f"observation space assumes {self.chunk_t}. Update "
                f"ACTION_CHUNK_T in experiments/tarc/config.py."
            )
        return chunk

    def _run(self):
        try:
            while not self._stop.is_set():
                # Timeout rather than a bare wait so _stop is noticed promptly
                # when no requests are coming.
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                if self._stop.is_set():
                    break
                with self._lock:
                    req, self._pending = self._pending, None
                    gen = self._gen
                if req is None:
                    continue
                chunk = self._infer(req)
                with self._lock:
                    self._busy = False
                    if self._gen == gen:
                        self._chunk = chunk
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread
            with self._lock:
                self._err = e
                self._busy = False
