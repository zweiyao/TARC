"""Stand-in for robot_servers/franka_server.py, for when there is no arm.

Same HTTP surface as the real server, but the arm is a variable. Stateful on
purpose: /pose writes the pose and /getstate reads it back, so interpolate_move
and go_to_reset actually converge instead of spinning against a constant.

    python examples/experiments/tarc/fake/franka_server.py

Only reachable when the caller points SERVER_URL at it; nothing imports this.
"""
import argparse

import numpy as np
from flask import Flask, request
from scipy.spatial.transform import Rotation as R

# Matches tarc/config.py's RESET_POSE. xyz + euler, converted to the xyz + quat
# the wire format uses.
RESET_POSE = np.array([0.588, -0.036, 0.328, np.pi, 0.05, 0.0])

app = Flask(__name__)


class Arm:
    def __init__(self):
        self.pose = np.concatenate(
            [RESET_POSE[:3], R.from_euler("xyz", RESET_POSE[3:]).as_quat()]
        )
        self.gripper = 1.0  # open; 0.85 is the threshold franka_env.py:427 uses


arm = Arm()


@app.route("/getstate", methods=["POST"])
def getstate():
    # The only route whose response is parsed (franka_env.py:444-455). Every
    # field below is read, and jacobian is reshaped to (6, 7) so its length is
    # load-bearing even though the observation never uses it.
    return {
        "pose": arm.pose.tolist(),
        "vel": np.zeros(6).tolist(),
        "force": np.zeros(3).tolist(),
        "torque": np.zeros(3).tolist(),
        "q": np.zeros(7).tolist(),
        "dq": np.zeros(7).tolist(),
        "jacobian": np.zeros(42).tolist(),
        # A scalar, not a list. As a list, curr_gripper_pos > 0.85 becomes an
        # array and the `and` in _send_gripper_command raises "truth value of an
        # array is ambiguous".
        "gripper_pos": float(arm.gripper),
    }


@app.route("/pose", methods=["POST"])
def pose():
    arm.pose = np.array(request.json["arr"])
    return "ok"


@app.route("/close_gripper", methods=["POST"])
@app.route("/close_gripper_slow", methods=["POST"])
def close_gripper():
    arm.gripper = 0.0
    return "ok"


@app.route("/open_gripper", methods=["POST"])
def open_gripper():
    arm.gripper = 1.0
    return "ok"


# Responses here are never read — franka_env.py does not even check the status
# code — so they only have to exist.
@app.route("/clearerr", methods=["POST"])
@app.route("/update_param", methods=["POST"])
@app.route("/jointreset", methods=["POST"])
@app.route("/set_load", methods=["POST"])
def ack():
    return "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5010)
    a = ap.parse_args()
    print(f"fake franka server on http://{a.host}:{a.port}/")
    app.run(host=a.host, port=a.port, threaded=True)
