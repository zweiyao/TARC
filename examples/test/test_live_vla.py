"""Exercises LiveActionChunkSource with a stubbed pi0.5 server.

Only client.infer() is faked. The thread, the lock, the 5:1 step cadence, the
backpressure, the generation check and pi05_client.to_request's real request
assembly all run as written. The served model is the one thing left uncovered.

    python examples/test/test_live_vla.py
"""
import sys
import threading
import time
from pathlib import Path

_here = Path(__file__).resolve()
sys.path[:0] = [str(_here.parents[1]), str(_here.parents[2])]

import numpy as np

from vla.live import LiveActionChunkSource, VLAStale

EVERY = 5
T = 15
DT = 0.02  # stand-in for the control period; the real one is ~0.107 s


class StubClient:
    """Returns a chunk filled with its own call index, so "which chunk am I
    holding" is directly observable. `latencies` is consumed one per call and
    the last value repeats."""

    def __init__(self, latencies=(0.01,)):
        self.latencies = list(latencies)
        self.calls = 0
        self.fail = None
        self.gate = None

    def infer(self, request):
        n = self.calls
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        if self.gate is not None:
            self.gate.wait()
        time.sleep(self.latencies[min(n, len(self.latencies) - 1)])
        return {"actions": np.full((T, 8), float(n), np.float32)}


def make_obs():
    """The pre-SERLObsWrapper dict: images nested, state still split."""
    return {
        "images": {
            "wrist_1": np.zeros((128, 128, 3), np.uint8),
            "wrist_2": np.zeros((128, 128, 3), np.uint8),
        },
        "state": {
            "tcp_pose": np.zeros(6),      # euler, downstream of Quat2Euler
            "tcp_vel": np.zeros(6),
            "tcp_force": np.zeros(3),
            "tcp_torque": np.zeros(3),
            "gripper_pose": np.zeros(1),  # 6+6+3+3+1 = STATE_DIM
        },
    }


def source(client, **kw):
    return LiveActionChunkSource(every=EVERY, chunk_t=T, state_dim=19,
                                 client=client, **kw)


def which(chunk):
    """The stub call index a published chunk came from."""
    return int(np.asarray(chunk)[0, 0])


# -- 1. shape, and one inference at construction ------------------------------

c = StubClient()
src = source(c)
chunk = src(None)  # the space probe TactileVLAWrapper.__init__ makes
assert chunk.shape == (T, 8) and chunk.dtype == np.float32, (chunk.shape, chunk.dtype)
assert c.calls == 1, f"__init__ must produce a chunk, got {c.calls} calls"
print(f"probe: {chunk.shape} {chunk.dtype}, {c.calls} inference at construction")

# -- 2. exactly one request per EVERY steps -----------------------------------

seen = []
for i in range(20):
    seen.append(which(src(make_obs())))
    time.sleep(DT)
time.sleep(0.1)  # let the last inference land

# steps 0, 5, 10, 15 fire; the probe is the +1
assert c.calls == 1 + 4, f"expected 4 requests over 20 steps, got {c.calls - 1}"
print(f"20 steps -> {c.calls - 1} inferences (want {20 // EVERY})")

# -- 3. the chunk only ever moves forward -------------------------------------

assert seen == sorted(seen), f"chunk went backwards: {seen}"
assert seen[-1] > seen[0], f"chunk never advanced: {seen}"
print(f"chunk index across 20 steps: {seen}")
src.close()

# -- 4. slow inference degrades the cadence instead of queueing, and the -----
#       control loop keeps paying nothing for it

c = StubClient(latencies=[0.01, 1.2])  # fast probe, then slower than EVERY steps
src = source(c)
slowest = 0.0
for _ in range(25):
    obs = make_obs()
    t0 = time.monotonic()
    src(obs)
    slowest = max(slowest, time.monotonic() - t0)
    time.sleep(DT)
# 25 steps x 20 ms = 0.5 s, well short of the 1.2 s inference, so the request
# issued at step 0 is still in flight and steps 5/10/15/20 must all skip.
assert c.calls == 2, f"requests queued up: {c.calls - 1} sent, expected 1"
print(f"25 steps with 1.2 s inference -> {c.calls - 1} request (no queueing)")
assert slowest < 0.05, f"a step blocked for {slowest * 1000:.1f} ms"
print(f"slowest __call__ meanwhile: {slowest * 1000:.2f} ms (inference is 1200 ms)")
src.close()

# -- 5. a wedged server raises instead of serving a frozen chunk --------------

c = StubClient(latencies=[0.01])
src = source(c, stale_s=0.3)
c.gate = threading.Event()  # every inference from here on blocks
deadline = time.time() + 5
while time.time() < deadline:
    try:
        src(make_obs())
    except VLAStale as e:
        print("stale detected:", str(e).splitlines()[0])
        break
    time.sleep(DT)
else:
    raise AssertionError("a wedged server was never reported")
c.gate.set()  # release the thread so close() can join
src.close()

# -- 6. a dead thread surfaces on the caller's thread -------------------------

c = StubClient()
src = source(c)
c.fail = RuntimeError("pi0.5 server died")
deadline = time.time() + 5
while time.time() < deadline:
    try:
        src(make_obs())
    except RuntimeError as e:
        assert "pi0.5 server died" in str(e), e
        print("thread error re-raised on the caller:", e)
        break
    time.sleep(DT)
else:
    raise AssertionError("a dead inference thread was never reported")
src.close()

# -- 7. refresh() publishes now, and a late in-flight result cannot win -------

# The step-0 request is slow (0.6 s); refresh's own inference is fast (0.05 s),
# so the stale result lands *after* the fresh one and the generation check is
# what has to reject it.
c = StubClient(latencies=[0.01, 0.6, 0.05])
src = source(c)
src(make_obs())            # step 0 fires the slow request (stub call 1)
time.sleep(0.05)
fresh = src.refresh(make_obs())   # stub call 2, lands first
assert which(fresh) == 2, which(fresh)
assert which(src(make_obs())) == 2, "refresh() did not publish"
time.sleep(0.8)            # the slow pre-refresh request lands in here
held = which(src(make_obs()))
assert held == 2, f"a pre-refresh chunk overwrote the fresh one: got {held}"
print("refresh() published, and the late pre-refresh result was discarded")
src.close()

# -- 8. refresh() restarts the cadence rather than firing again immediately ---

c = StubClient()
src = source(c)
src.refresh(make_obs())
before = c.calls
for _ in range(EVERY - 1):  # refresh() counts as the firing step
    src(make_obs())
    time.sleep(DT)
assert c.calls == before, f"fired early: {c.calls - before} extra requests"
src(make_obs())             # the EVERY-th step after refresh
time.sleep(0.2)
assert c.calls == before + 1, f"cadence did not restart: {c.calls - before}"
print(f"refresh() restarted the {EVERY}-step cadence")
src.close()

print("OK")
