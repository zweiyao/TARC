import copy
import time
from franka_env.utils.rotations import euler_2_quat
from scipy.spatial.transform import Rotation as R
import numpy as np
import gymnasium as gym
import requests

from franka_env.envs.franka_env import FrankaEnv


class TarcEnv(FrankaEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.should_regrasp = False

        # Imported here rather than at module scope: ram_insertion/wrapper.py:7
        # does it at the top, so merely importing that task fails on a headless
        # box, before anything gets a chance to decide it does not need a
        # keyboard.
        from pynput import keyboard

        def on_press(key):
            if str(key) == "Key.f1":
                self.should_regrasp = True

        listener = keyboard.Listener(
            on_press=on_press)
        listener.start()

    def go_to_reset(self, joint_reset=False):
        """
        Move to the rest position defined in base class.
        Add a small z offset before going to rest to avoid collision with object.
        """        
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        # perform joint reset if needed
        if joint_reset:
            print("JOINT RESET")
            requests.post(self.url + "jointreset")
            time.sleep(0.5)

        # perform Cartesian reset
        if self.randomreset:  # randomize reset position in xy plane
            reset_pose = self.resetpos.copy()
            reset_pose[:2] += np.random.uniform(
                -self.random_xy_range, self.random_xy_range, (2,)
            )
            euler_random = self._RESET_POSE[3:].copy()
            euler_random[-1] += np.random.uniform(
                -self.random_rz_range, self.random_rz_range
            )
            reset_pose[3:] = euler_2_quat(euler_random)
            self._send_pos_command(reset_pose)
        else:
            reset_pose = self.resetpos.copy()
            self._send_pos_command(reset_pose)
        time.sleep(0.5)

        # Change to compliance mode
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)


    def regrasp(self):
        # use compliance mode for coupled reset
        self._update_currpos()
        self._send_pos_command(self.currpos)
        time.sleep(0.3)
        requests.post(self.url + "update_param", json=self.config.PRECISION_PARAM)

        # pull up
        self._update_currpos()
        reset_pose = copy.deepcopy(self.currpos)
        reset_pose[2] = self.resetpos[2] + 0.04
        self.interpolate_move(reset_pose, timeout=1)

        input("Press enter to release gripper...")
        self._send_gripper_command(1.0)
        input("Place RAM in holder and press enter to grasp...")
        top_pose = self.config.GRASP_POSE.copy()
        top_pose[2] += 0.05
        top_pose[0] += np.random.uniform(-0.005, 0.005)
        self.interpolate_move(top_pose, timeout=1)
        time.sleep(0.5)

        grasp_pose = top_pose.copy()
        grasp_pose[2] -= 0.05
        self.interpolate_move(grasp_pose, timeout=0.5)

        requests.post(self.url + "close_gripper_slow")
        self.last_gripper_act = time.time()
        time.sleep(2)

        self.interpolate_move(top_pose, timeout=0.5)
        time.sleep(0.2)

        self.interpolate_move(self.config.RESET_POSE, timeout=1)
        time.sleep(0.5)


    def reset(self, joint_reset=False, **kwargs):
        self.last_gripper_act = time.time()
        if self.save_video:
            self.save_video_recording()

        # if True:
        if self.should_regrasp:
            self.regrasp()
            self.should_regrasp = False

        self._recover()
        self.go_to_reset(joint_reset=False)
        self._recover()
        self.curr_path_length = 0

        self._update_currpos()
        obs = self._get_obs()
        requests.post(self.url + "update_param", json=self.config.COMPLIANCE_PARAM)
        self.terminate = False
        return obs, {}

class TactileVLAWrapper(gym.ObservationWrapper):
    """Adds the tactile heatmap and the VLA reference action chunk to the obs.

    Sits before SERLObsWrapper, so it works on the raw
    {"images": {...}, "state": {...}} dict the env produces.

    The two land in different places because they are different things:

    - the tactile heatmap goes into obs["images"], since it *is* an image and
      SERLObsWrapper lifts everything in there to the top level for free. It
      must still be kept out of TrainConfig.image_keys, which is what drives the
      crop augmentation and the replay buffer's frame reuse.
    - the action chunk becomes its own top-level key. It is not an image, and
      putting it in obs["images"] would make every consumer that iterates images
      trip over it. This is what SERLObsWrapper had to learn to pass through.

    Both sources are injected as callables so switching from the synthetic
    placeholders to the real sensor and a live pi0.5 touches only config.py.
    """

    def __init__(self, env, tactile_fn, action_chunk_fn, tactile_key="tactile",
                 action_chunk_key="action_chunk"):
        super().__init__(env)
        self.tactile_fn = tactile_fn
        self.action_chunk_fn = action_chunk_fn
        self.tactile_key = tactile_key
        self.action_chunk_key = action_chunk_key

        # Probe both sources once so the spaces come from the same code that
        # will fill them, rather than from hardcoded shapes that could drift.
        tactile = np.asarray(tactile_fn())
        chunk = np.asarray(action_chunk_fn(None))

        spaces = dict(env.observation_space.spaces)
        images = gym.spaces.Dict(
            {
                **dict(spaces["images"].spaces),
                tactile_key: gym.spaces.Box(0, 255, tactile.shape, tactile.dtype),
            }
        )
        self.observation_space = gym.spaces.Dict(
            {
                **spaces,
                "images": images,
                action_chunk_key: gym.spaces.Box(
                    -np.inf, np.inf, chunk.shape, chunk.dtype
                ),
            }
        )

    def observation(self, obs):
        obs["images"][self.tactile_key] = self.tactile_fn()
        obs[self.action_chunk_key] = self.action_chunk_fn(obs)
        return obs
