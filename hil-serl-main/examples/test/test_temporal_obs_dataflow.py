"""Checks the extra time-series observation input, fed by pi0.5.

Builds an observation by hand, asks the pi0.5 server for an action chunk, feeds
that chunk in as the time-series input, runs the policy to get an action, then
runs one update() step. No robot, no hil-serl server, no demo data.

Needs the pi0.5 policy server running first:  bash vla/serve_pi05.sh

    python -m examples.test.test_temporal_obs_dataflow   # from hil-serl-main/
"""
import numpy as np
import jax
from flax.core import frozen_dict

from serl_launcher.utils.launcher import make_sac_pixel_agent
from vla import pi05_client

IMAGE_KEYS = ["wrist_1", "wrist_2"]
STATE_DIM = 19  # tcp_pose 6 + tcp_vel 6 + tcp_force 3 + tcp_torque 3 + gripper 1
ACTION_DIM = 6
TEMPORAL_KEY = "series"
BATCH = 4


def base_obs(batch=None):
    """Same shapes the wrapper chain would produce; see smoke_learner.py."""
    lead = (1,) if batch is None else (batch, 1)
    return {
        **{k: np.zeros(lead + (128, 128, 3), np.uint8) for k in IMAGE_KEYS},
        "state": np.zeros(lead + (STATE_DIM,), np.float32),
    }


# The chunk's own shape drives every series below, so a mismatch between the
# config's action_horizon and what the model actually returns cannot desync the
# agent from the data.
CHUNK = pi05_client.infer(pi05_client.connect(), base_obs())
print("vla chunk:", CHUNK.shape, CHUNK.dtype)


def make_obs(batch=None):
    obs = base_obs(batch)
    lead = obs["state"].shape[:-1]
    obs[TEMPORAL_KEY] = np.broadcast_to(CHUNK, lead + CHUNK.shape).astype(np.float32)
    return obs


agent = make_sac_pixel_agent(
    seed=0,
    sample_obs=make_obs(),
    sample_action=np.zeros((ACTION_DIM,), np.float32),
    image_keys=IMAGE_KEYS,
    temporal_key=TEMPORAL_KEY,
)

action = agent.sample_actions(make_obs(), seed=jax.random.PRNGKey(0), argmax=True)
assert action.shape == (ACTION_DIM,), action.shape
print("action:", np.asarray(action).round(3))

# update() needs a FrozenDict: data_augmentation_fn calls .copy(add_or_replace=...)
batch = frozen_dict.freeze(
    {
        "observations": make_obs(BATCH),
        "next_observations": make_obs(BATCH),
        "actions": np.zeros((BATCH, ACTION_DIM), np.float32),
        "rewards": np.zeros((BATCH,), np.float32),
        "masks": np.ones((BATCH,), np.float32),
    }
)
agent, info = agent.update(batch)
print("critic_loss:", round(float(info["critic"]["critic_loss"]), 4))
print("OK")
