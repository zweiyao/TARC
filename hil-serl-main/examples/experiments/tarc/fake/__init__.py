"""In-process stand-ins for the hardware franka_env needs but a dev box lacks.

install() must run BEFORE franka_env is imported: the camera and SpaceMouse
drivers are pulled in at module scope (rs_capture.py:2, pyspacemouse.py:1), so
by the time the env class exists it is already too late to swap them.

Gated on TARC_FAKE_SOURCES so none of this is reachable — not even importable —
on a real robot. The HTTP arm is a separate process, see fake/franka_server.py.
"""
import os
import sys
import types

import numpy as np

# Bigger than the largest IMAGE_CROP slice, since get_im() crops before it
# resizes. Both cameras claim (1280, 720) in the config.
FRAME_SHAPE = (720, 1280, 3)


class FakeRSCapture:
    """Duck-types RSCapture: .name, .read() -> (ok, frame), .close()."""

    def __init__(self, name, serial_number=None, dim=None, exposure=None, **kw):
        self.name = name
        # Deterministic noise rather than zeros: a constant image makes every
        # ResNet embedding identical, which would hide a wiring mistake behind a
        # test that still passes.
        rng = np.random.default_rng(abs(hash(name)) % (2**32))
        self._frame = rng.integers(0, 256, FRAME_SHAPE, dtype=np.uint8)

    def read(self):
        # Never block. get_im()'s queue.Empty path (franka_env.py:270-276) calls
        # input() and waits for a human, which would hang an unattended run.
        return True, self._frame.copy()

    def close(self):
        pass


def _stub_pyrealsense2():
    if "pyrealsense2" in sys.modules:
        return
    m = types.ModuleType("pyrealsense2")
    for name in ("pipeline", "config", "stream", "format", "context", "option"):
        setattr(m, name, types.SimpleNamespace())
    sys.modules["pyrealsense2"] = m


def _stub_pynput():
    """Replace pynput outright rather than using PYNPUT_BACKEND=dummy.

    The dummy backend imports fine but its Listener.start() raises
    NotImplementedError on a background thread, which prints an ownerless
    traceback and leaves the ESC-to-terminate hook silently dead. A stub is both
    quieter and more honest: nothing pretends to listen.
    """
    if "pynput" in sys.modules:
        return

    class _Listener:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            pass

        def stop(self):
            pass

        def join(self, *a, **kw):
            pass

    keyboard = types.SimpleNamespace(
        Listener=_Listener,
        Key=types.SimpleNamespace(esc="Key.esc", f1="Key.f1"),
    )
    m = types.ModuleType("pynput")
    m.keyboard = keyboard
    # Marked so KeyboardRewardWrapper can tell "nobody is listening" from
    # "nobody pressed a key" — behaviourally those are identical, and guessing
    # would mean silently reporting reward 0 forever on a machine where the
    # keyboard actually works.
    m._TARC_STUB = True
    sys.modules["pynput"] = m
    sys.modules["pynput.keyboard"] = keyboard


def install():
    """Swap in the fakes. Idempotent; safe to call more than once."""
    _stub_pynput()
    _stub_pyrealsense2()

    from franka_env.camera import rs_capture

    rs_capture.RSCapture = FakeRSCapture
    # franka_env.py:395 resolves RSCapture through its own module namespace, so
    # patching only the defining module would leave the real class in play.
    from franka_env.envs import franka_env as franka_env_mod

    if hasattr(franka_env_mod, "RSCapture"):
        franka_env_mod.RSCapture = FakeRSCapture

    from franka_env.spacemouse import spacemouse_expert

    class FakeSpaceMouseExpert:
        def get_action(self):
            # Below the 0.001 norm threshold at wrappers.py:230, so the wrapper
            # reads it as "no human is intervening" and leaves the policy action
            # untouched. Exactly two buttons: wrappers.py:227 unpacks them into
            # (left, right), and a longer list raises there.
            return np.zeros(6), [0, 0]

        def close(self):
            pass

    spacemouse_expert.SpaceMouseExpert = FakeSpaceMouseExpert
    from franka_env.envs import wrappers as env_wrappers

    if hasattr(env_wrappers, "SpaceMouseExpert"):
        env_wrappers.SpaceMouseExpert = FakeSpaceMouseExpert

    print("[tarc.fake] cameras, spacemouse and pynput stubbed")
