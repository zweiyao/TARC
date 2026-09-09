"""Exercises LiveTactileSource over real shared memory, with no camera.

shmframe.py ships a FrameWriter, so the segment, the reader, the background
thread, the baseline and the stale detection are all the real code paths — only
the frame content is synthetic. The camera driver itself is the one thing left
uncovered.

    python examples/test/test_live_tactile.py
"""
import sys
import threading
import time
from pathlib import Path

_here = Path(__file__).resolve()
sys.path[:0] = [str(_here.parents[1]), str(_here.parents[2])]

import numpy as np

from tactile import synth
from tactile.live import LiveTactileSource, TactileStale
from tactile.shmframe import FrameWriter

SHM = "tactile_test_frames"
H, W = 1080, 1920


class Camera:
    """Publishes speckle frames at ~30 Hz, like capture.py does."""

    def __init__(self, name=SHM, hz=30.0):
        self.w = FrameWriter(name, H, W, slots=8)
        self.base = synth.speckle((H, W), seed=0)
        self.peak = 0.0
        self.hz = hz
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set():
            frame = self.base if self.peak == 0 else synth.warp(self.base, self.peak)[0]
            self.w.publish(frame, time.time(), time.monotonic())
            time.sleep(1.0 / self.hz)

    def pause(self):
        self._stop.set()
        self._t.join(timeout=2.0)

    def close(self):
        self.pause()
        self.w.close()


cam = Camera()
time.sleep(0.3)  # let a few frames land so the baseline has something to average

# 3 reference frames rather than the production 30: read_distinct pulls them at
# camera rate, and this is checking the wiring, not the noise floor.
src = LiveTactileSource(shm_name=SHM, ref_frames=3, stale_s=2.0)

heat = src()
assert heat.shape == (240, 320, 3) and heat.dtype == np.uint8, (heat.shape, heat.dtype)
print("heatmap:", heat.shape, heat.dtype)

# Against its own baseline the gel reads as untouched, so the heatmap should sit
# at the bottom of the TURBO ramp.
rest = float(np.asarray(heat, np.float32).mean())

# Now press. The heatmap has to follow — a frozen image would pass every shape
# check while carrying nothing, which is exactly what the synthetic placeholder
# does today.
cam.peak = 2.5
time.sleep(0.6)
pressed = src()
press_mean = float(np.asarray(pressed, np.float32).mean())
assert not np.array_equal(heat, pressed), "heatmap did not change under contact"
assert press_mean > rest, (rest, press_mean)
print(f"rest mean {rest:.1f} -> pressed mean {press_mean:.1f}")

# Stop publishing: the reader must raise rather than keep serving a stale image.
cam.pause()
deadline = time.time() + 8
while time.time() < deadline:
    try:
        src()
    except TactileStale as e:
        print("stale detected:", str(e)[:60])
        break
    time.sleep(0.2)
else:
    raise AssertionError("stale frames were served indefinitely")

# Order matters: the reader calls resource_tracker.unregister on the segment
# (shmframe.py:71, so a reader exiting cannot unlink the writer's memory), and
# the writer's unlink then has nothing left to deregister. Dropping the writer
# first keeps the tracker quiet.
cam.close()
src.close()
print("OK")
