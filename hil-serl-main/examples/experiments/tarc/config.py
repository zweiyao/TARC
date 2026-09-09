import os
import jax
import jax.numpy as jnp
import numpy as np

from franka_env.envs.wrappers import (
    Quat2EulerWrapper,
    SpacemouseIntervention,
    MultiCameraBinaryRewardClassifierWrapper,
    GripperCloseEnv
)
from franka_env.envs.relative_env import RelativeFrame
from franka_env.envs.franka_env import DefaultEnvConfig
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper
from serl_launcher.networks.reward_classifier import load_classifier_func

from experiments.config import DefaultTrainingConfig

# Set by the *_fake.sh launchers. The real entry point never sets it, so none of
# the synthetic paths below are reachable on hardware.
# `TARC_FAKE_SOURCES=0` reads as "off" the way a shell user expects; bool() on
# the raw string would make it true.
FAKE_SOURCES = os.environ.get("TARC_FAKE_SOURCES", "") not in ("", "0", "false")

# tcp_pose(6, euler) + tcp_vel(6) + tcp_force(3) + tcp_torque(3) + gripper(1).
# Holds only downstream of Quat2EulerWrapper, which is why TactileVLAWrapper
# sits after it.
STATE_DIM = 19

# pi05_droid's action_horizon. The learner never connects to pi0.5, so its
# observation space has to assume a T; the actor asserts the served model agrees
# rather than letting the mismatch surface as a numpy broadcast error on the
# learner's first demo_buffer.insert.
ACTION_CHUNK_T = 15
from experiments.tarc.wrapper import (
    TarcEnv,
    TactileVLAWrapper,
    KeyboardRewardWrapper,
)

class EnvConfig(DefaultEnvConfig):
    # The real franka_server binds --flask_url on port 5000
    # (robot_servers/launch_right_server.sh:13). fake/franka_server.py uses
    # 127.0.0.1:5010 instead: the .2 alias needs `ifconfig lo0 alias` on macOS,
    # and 5000 is a coin flip on a shared box.
    SERVER_URL = (
        "http://127.0.0.1:5010/" if FAKE_SOURCES else "http://127.0.0.2:5000/"
    )
    REALSENSE_CAMERAS = {
        "wrist_1": {
            "serial_number": "127122270146",
            "dim": (1280, 720),
            "exposure": 40000,
        },
        "wrist_2": {
            "serial_number": "127122270350",
            "dim": (1280, 720),
            "exposure": 40000,
        },
    }
    IMAGE_CROP = {
        "wrist_1": lambda img: img[150:450, 350:1100],
        "wrist_2": lambda img: img[100:500, 400:900],
    }
    TARGET_POSE = np.array([0.5881241235410154,-0.03578590131997776,0.27843494179085326, np.pi, 0, 0])
    GRASP_POSE = np.array([0.5857508505445138,-0.22036261105675414,0.2731021902359492, np.pi, 0, 0])
    RESET_POSE = TARGET_POSE + np.array([0, 0, 0.05, 0, 0.05, 0])
    ABS_POSE_LIMIT_LOW = TARGET_POSE - np.array([0.03, 0.02, 0.01, 0.01, 0.1, 0.4])
    ABS_POSE_LIMIT_HIGH = TARGET_POSE + np.array([0.03, 0.02, 0.05, 0.01, 0.1, 0.4])
    RANDOM_RESET = True
    RANDOM_XY_RANGE = 0.02
    RANDOM_RZ_RANGE = 0.05
    ACTION_SCALE = (0.01, 0.06, 1)
    # ImageDisplayer calls cv2.imshow on a background thread; with no display
    # it throws there, where nothing catches it. The real workstation has one.
    DISPLAY_IMAGE = not FAKE_SOURCES
    MAX_EPISODE_LENGTH = 100
    COMPLIANCE_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 150,
        "rotational_damping": 7,
        "translational_Ki": 0,
        "translational_clip_x": 0.0075,
        "translational_clip_y": 0.0016,
        "translational_clip_z": 0.0055,
        "translational_clip_neg_x": 0.002,
        "translational_clip_neg_y": 0.0016,
        "translational_clip_neg_z": 0.005,
        "rotational_clip_x": 0.01,
        "rotational_clip_y": 0.025,
        "rotational_clip_z": 0.005,
        "rotational_clip_neg_x": 0.01,
        "rotational_clip_neg_y": 0.025,
        "rotational_clip_neg_z": 0.005,
        "rotational_Ki": 0,
    }
    PRECISION_PARAM = {
        "translational_stiffness": 2000,
        "translational_damping": 89,
        "rotational_stiffness": 250,
        "rotational_damping": 9,
        "translational_Ki": 0.0,
        "translational_clip_x": 0.1,
        "translational_clip_y": 0.1,
        "translational_clip_z": 0.1,
        "translational_clip_neg_x": 0.1,
        "translational_clip_neg_y": 0.1,
        "translational_clip_neg_z": 0.1,
        "rotational_clip_x": 0.5,
        "rotational_clip_y": 0.5,
        "rotational_clip_z": 0.5,
        "rotational_clip_neg_x": 0.5,
        "rotational_clip_neg_y": 0.5,
        "rotational_clip_neg_z": 0.5,
        "rotational_Ki": 0.0,
    }


def _tactile_source(spaces_only=False):
    """() -> (240, 320, 3) uint8 RGB heatmap, called once per env step."""
    if spaces_only:
        blank = np.zeros((240, 320, 3), np.uint8)
        return lambda: blank

    if not FAKE_SOURCES:
        # Reads capture.py's shared memory on a background thread so the ~31 ms
        # of Farneback stays off the control loop; see tactile/live.py.
        from tactile.live import LiveTactileSource

        return LiveTactileSource()

    # PLACEHOLDER with no sensor attached: synthetic speckle through the real
    # flow.py pipeline. Note this returns one frozen image — the flow is hoisted
    # out of step() to save 30 ms, which is fine for a dataflow check but means
    # the tactile channel carries no information.
    from tactile import synth

    ref, cur, _ = synth.press(peak_px=2.5)
    heat = synth.heatmap(cur, ref)
    return lambda: heat


def _action_chunk_source(spaces_only=False):
    """(obs) -> (T, 8) float32 reference action chunk from pi0.5.

    T comes from the served checkpoint (15 for pi05_droid); callers read it off
    the returned array rather than assuming.
    """
    if spaces_only:
        # Shape only — the learner never reads these values, and connecting
        # would make it wait on a service it has no use for.
        blank = np.zeros((ACTION_CHUNK_T, 8), np.float32)
        return lambda obs: blank

    from vla import pi05_client

    client = pi05_client.connect()
    zeros = np.zeros((1, 128, 128, 3), np.uint8)

    def infer(obs):
        if obs is None:  # probe call from TactileVLAWrapper, for the space only
            state = np.zeros((1, STATE_DIM), np.float32)
            frame = zeros
        else:
            state = np.concatenate(
                [np.atleast_1d(obs["state"][k]).ravel() for k in
                 ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")]
            )[None].astype(np.float32)
            # The probe above claims STATE_DIM. If the two ever disagree the
            # request silently carries different numbers than the space says,
            # and pi05_client's slicing lands on the wrong fields.
            assert state.shape[1] == STATE_DIM, (state.shape, STATE_DIM)
            frame = None
        req = {
            "wrist_1": frame if frame is not None else obs["images"]["wrist_1"][None],
            "wrist_2": frame if frame is not None else obs["images"]["wrist_2"][None],
            "state": state,
        }
        chunk = pi05_client.infer(client, req).astype(np.float32)
        assert chunk.shape[0] == ACTION_CHUNK_T, (
            f"pi0.5 returned T={chunk.shape[0]}, but the learner's observation "
            f"space assumes {ACTION_CHUNK_T}. Update ACTION_CHUNK_T."
        )
        return chunk

    return infer


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    # Deliberately not in image_keys: that list drives the crop augmentation
    # (launcher.py:212) and the replay buffer's frame reuse, neither of which
    # suits tactile.
    tactile_keys = ["tactile"]
    action_chunk_key = "action_chunk"
    # tactile is outside image_keys (that list drives the crop augmentation),
    # but train_rlpd_tarc.py hands it to the buffer anyway, so it does get frame
    # reuse and is stored once rather than twice. Even so it dominates: at
    # 240x320x3 it is 230 KB per step against 48 KB for each camera, so 50k
    # steps is ~16 GB per buffer and the learner builds two.
    replay_buffer_capacity = 2000 if FAKE_SOURCES else 50000
    classifier_keys = ["wrist_1", "wrist_2"]
    proprio_keys = ["tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose"]
    buffer_period = 1000
    checkpoint_period = 5000
    steps_per_update = 50
    encoder_type = "resnet-pretrained"
    setup_mode = "single-arm-fixed-gripper"

    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        env = TarcEnv(
            fake_env=fake_env,
            save_video=save_video,
            config=EnvConfig(),
        )
        env = GripperCloseEnv(env)
        if not fake_env:
            env = SpacemouseIntervention(env)
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        # After Quat2EulerWrapper on purpose: tcp_pose is 6-dim euler here, so
        # the state this wrapper hands to pi0.5 is the same 19 dims the policy
        # eventually sees. Before it, tcp_pose is still a 7-dim quaternion and
        # the totals silently disagree (20 vs the 19 the probe builds).
        # Still before SERLObsWrapper, which is what lifts obs["images"] to the
        # top level and would otherwise drop the action chunk.
        env = TactileVLAWrapper(
            env,
            # fake_env means the learner process, which only needs the spaces.
            # Building the real sources there would make the learner depend on
            # capture.py, pi0.5 and the arm for data it never reads, and leave a
            # flow thread running with nobody to surface its errors.
            tactile_fn=_tactile_source(fake_env),
            action_chunk_fn=_action_chunk_source(fake_env),
            action_chunk_key=self.action_chunk_key,
        )
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # A human at the keyboard judges success instead of a trained
        # classifier, so there is no classifier_ckpt/ to collect data for and
        # train first. `classifier` is kept in the signature because
        # DefaultTrainingConfig declares it and the other entry points pass it,
        # but it no longer selects anything.
        env = KeyboardRewardWrapper(env)
        return env