"""π0.5 (openpi pi05_droid) inference client.

The model runs in a separate process and conda env — openpi needs Python 3.11
and jax 0.5.3, which would break this env's pinned jax 0.4.35. Only the
lightweight openpi-client package is installed here; everything else goes over
a websocket. See vla/README.md for setup.
"""
import urllib.request

import numpy as np
from openpi_client import image_tools, websocket_client_policy

# Port 8010, not openpi's default 8000 — that one is taken on the shared box.
HOST, PORT = "127.0.0.1", 8010

# PLACEHOLDER. hil-serl only has two wrist cameras; DROID expects one exterior
# view plus one wrist view. Mapped by position, not by meaning. Change these two
# lines once an exterior camera exists.
BASE_CAM, WRIST_CAM = "wrist_1", "wrist_2"

# PLACEHOLDER. Change when the real task is wired up.
PROMPT = "pick up the usb lamp"


def _frame(x, ndim):
    """Drop the obs-horizon axis that ChunkingWrapper adds.

    DroidInputs._parse_image only transposes when shape[0] == 3, so passing
    (1, 224, 224, 3) sails through and breaks further downstream.
    """
    x = np.asarray(x)
    assert x.ndim == ndim and x.shape[0] == 1, f"expected a single frame, got {x.shape}"
    return x[0]


def to_request(obs, prompt=PROMPT):
    """hil-serl observation -> pi05_droid request.

    PLACEHOLDER MAPPING. hil-serl's 19-dim state is
    tcp_pose(6) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper(1) — it
    carries no joint angles at all, which is exactly what pi05_droid wants. The
    tcp_pose is dropped into the first six joint slots and the seventh is zero,
    purely so the data flows. The numbers mean nothing physically. The gripper
    is the one quantity that lines up on both sides.
    """
    state = _frame(obs["state"], 2)
    joint = np.zeros(7, np.float32)
    joint[:6] = state[:6]
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            _frame(obs[BASE_CAM], 4), 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            _frame(obs[WRIST_CAM], 4), 224, 224
        ),
        "observation/joint_position": joint,
        "observation/gripper_position": state[-1:].astype(np.float32),
        "prompt": prompt,
    }


def connect(host=HOST, port=PORT):
    """Connect to the policy server, failing fast with something actionable.

    WebsocketClientPolicy retries a refused connection every 5s forever, so
    without this probe a missing server just hangs silently.
    """
    try:
        urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=2).close()
    except Exception as e:
        raise RuntimeError(
            f"no pi0.5 server on {host}:{port} ({type(e).__name__}: {e})\n"
            f"  start it:  bash vla/serve_pi05.sh\n"
            f"  wait:      curl -s -o /dev/null -w '%{{http_code}}\\n' "
            f"http://{host}:{port}/healthz\n"
            f"  first run also needs the checkpoint — see vla/README.md"
        ) from None
    return websocket_client_policy.WebsocketClientPolicy(host, port)


def infer(client, obs, prompt=PROMPT):
    """Return a (T, 8) action chunk: 7 joint velocities + 1 gripper.

    T comes from the model — callers must not hardcode it.
    """
    chunk = np.asarray(client.infer(to_request(obs, prompt))["actions"], np.float32)
    assert chunk.ndim == 2, f"expected (T, C), got {chunk.shape}"
    return chunk
