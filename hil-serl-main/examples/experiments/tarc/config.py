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
from experiments.tarc.wrapper import (
    TarcEnv,
    TactileVLAWrapper,
    KeyboardRewardWrapper,
)

class EnvConfig(DefaultEnvConfig):
    # 5010 on loopback, not the 127.0.0.2:5000 ram_insertion uses: that alias
    # needs `ifconfig lo0 alias` on macOS, and this box is shared so a common
    # port is a coin flip. fake/franka_server.py defaults to the same.
    SERVER_URL = "http://127.0.0.1:5010/"
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
    # it throws in that thread where nothing catches it.
    DISPLAY_IMAGE = False
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


def _tactile_source():
    """() -> (240, 320, 3) uint8 heatmap.

    PLACEHOLDER while the sensor is unplugged: synthetic speckle through the
    real flow.py pipeline. The live version reads frames off capture.py's shared
    memory and calls the same tactile.synth.heatmap.
    """
    from tactile import synth

    ref, cur, _ = synth.press(peak_px=2.5)
    heat = synth.heatmap(cur, ref)  # ~30ms of Farneback; hoisted out of step()
    return lambda: heat


def _action_chunk_source():
    """(obs) -> (15, 8) float32 reference action chunk from pi0.5."""
    from vla import pi05_client

    client = pi05_client.connect()
    zeros = np.zeros((1, 128, 128, 3), np.uint8)

    def infer(obs):
        if obs is None:  # probe call from TactileVLAWrapper, for the space only
            state = np.zeros((1, 19), np.float32)
            frame = zeros
        else:
            state = np.concatenate(
                [np.atleast_1d(obs["state"][k]).ravel() for k in
                 ("tcp_pose", "tcp_vel", "tcp_force", "tcp_torque", "gripper_pose")]
            )[None].astype(np.float32)
            frame = None
        req = {
            "wrist_1": frame if frame is not None else obs["images"]["wrist_1"][None],
            "wrist_2": frame if frame is not None else obs["images"]["wrist_2"][None],
            "state": state,
        }
        return pi05_client.infer(client, req).astype(np.float32)

    return infer


class TrainConfig(DefaultTrainingConfig):
    image_keys = ["wrist_1", "wrist_2"]
    # Deliberately not in image_keys: that list drives the crop augmentation
    # (launcher.py:212) and the replay buffer's frame reuse, neither of which
    # suits tactile.
    tactile_keys = ["tactile"]
    action_chunk_key = "action_chunk"
    # tactile is deliberately outside image_keys, so the buffer keeps a full
    # copy in both observations and next_observations rather than reusing
    # frames. At the 200000 default that is ~92 GB of np.empty for the tactile
    # channel alone, times four buffers across actor and learner. Raise this
    # when a real arm is attached and the tactile resolution is settled.
    replay_buffer_capacity = 2000
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
        env = TactileVLAWrapper(
            env,
            tactile_fn=_tactile_source(),
            action_chunk_fn=_action_chunk_source(),
            action_chunk_key=self.action_chunk_key,
        )
        env = RelativeFrame(env)
        env = Quat2EulerWrapper(env)
        env = SERLObsWrapper(env, proprio_keys=self.proprio_keys)
        env = ChunkingWrapper(env, obs_horizon=1, act_exec_horizon=None)
        # A human at the keyboard judges success instead of a trained
        # classifier, so there is no classifier_ckpt/ to collect data for and
        # train first. `classifier` is kept in the signature because
        # DefaultTrainingConfig declares it and the other entry points pass it,
        # but it no longer selects anything.
        env = KeyboardRewardWrapper(env)
        return env