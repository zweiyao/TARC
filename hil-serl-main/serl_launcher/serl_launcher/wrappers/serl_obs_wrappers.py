import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class SERLObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper treat the observation space as a dictionary
    of a flattened state space and the images.
    """

    def __init__(self, env, proprio_keys=None):
        super().__init__(env)
        self.proprio_keys = proprio_keys
        if self.proprio_keys is None:
            self.proprio_keys = list(self.env.observation_space["state"].keys())

        self.proprio_space = gym.spaces.Dict(
            {key: self.env.observation_space["state"][key] for key in self.proprio_keys}
        )

        # Anything that is neither "state" nor "images" used to be dropped here
        # without a word, so a non-image modality could be added upstream and
        # simply never arrive. Carry it through untouched instead.
        self.extra_keys = [
            k for k in self.env.observation_space.spaces if k not in ("state", "images")
        ]

        self.observation_space = gym.spaces.Dict(
            {
                "state": flatten_space(self.proprio_space),
                **(self.env.observation_space["images"]),
                **{k: self.env.observation_space[k] for k in self.extra_keys},
            }
        )

    def observation(self, obs):
        obs = {
            "state": flatten(
                self.proprio_space,
                {key: obs["state"][key] for key in self.proprio_keys},
            ),
            **(obs["images"]),
            **{k: obs[k] for k in self.extra_keys},
        }
        return obs

    def reset(self, **kwargs):
        obs, info =  self.env.reset(**kwargs)
        return self.observation(obs), info

def flatten_observations(obs, proprio_space, proprio_keys):
        # Mirrors SERLObsWrapper.observation for replayed transitions; it has to
        # keep the extra keys too or replay silently disagrees with rollout.
        extra = {k: v for k, v in obs.items() if k not in ("state", "images")}
        obs = {
            "state": flatten(
                proprio_space,
                {key: obs["state"][key] for key in proprio_keys},
            ),
            **(obs["images"]),
            **extra,
        }
        return obs