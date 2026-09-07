"""Checks the extra time-series observation input, fed by pi0.5.

Builds an observation by hand, asks the pi0.5 server for an action chunk, feeds
that chunk in as the time-series input, runs the policy to get an action, then
runs one update() step. No robot, no hil-serl server, no demo data.

Needs the pi0.5 policy server running first:  bash vla/serve_pi05.sh

    python -m examples.test.test_temporal_obs_dataflow   # from hil-serl-main/
    python examples/test/test_temporal_obs_dataflow.py   # works too
"""
import sys
from pathlib import Path

# Running this by file path puts examples/test/ on sys.path rather than the repo
# root, so `vla` would not resolve. Put the repo root on the path ourselves so
# every invocation style works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import gymnasium as gym
import numpy as np
import jax
from flax.core import frozen_dict

from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.launcher import make_sac_pixel_agent
from vla import pi05_client

IMAGE_KEYS = ["wrist_1", "wrist_2"]
# Tactile shares the frozen ResNet trunk with the cameras but is kept out of
# image_keys, which is what drives the crop augmentation (launcher.py:212) and
# the replay buffer's frame reuse (memory_efficient_replay_buffer.py:40). Its
# encoder head keeps the native 240x320 and skips the ImageNet mean shift.
TACTILE_KEYS = ["tactile"]
OBS_SHAPES = {
    "wrist_1": (128, 128, 3),
    "wrist_2": (128, 128, 3),
    "tactile": (240, 320, 3),
}
STATE_DIM = 19  # tcp_pose 6 + tcp_vel 6 + tcp_force 3 + tcp_torque 3 + gripper 1
ACTION_DIM = 6
TEMPORAL_KEY = "series"
BATCH = 4


def base_obs(batch=None):
    """Same shapes the wrapper chain would produce; see smoke_learner.py."""
    lead = (1,) if batch is None else (batch, 1)
    return {
        **{k: np.zeros(lead + OBS_SHAPES[k], np.uint8) for k in IMAGE_KEYS + TACTILE_KEYS},
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


def base_obs_with_series():
    """One unbatched transition for the buffer, which stores per-step frames."""
    return {k: v[0] for k, v in make_obs().items()}


agent = make_sac_pixel_agent(
    seed=0,
    sample_obs=make_obs(),
    sample_action=np.zeros((ACTION_DIM,), np.float32),
    image_keys=IMAGE_KEYS,
    tactile_keys=TACTILE_KEYS,
    temporal_key=TEMPORAL_KEY,
)

p = agent.state.params["modules_actor"]
enc = p["encoder"]

# The trunk must exist exactly once. Flax stores it under whichever encoder_*
# key sorts first; two copies would mean the sharing broke.
trunks = [n for n, h in enc.items() if hasattr(h, "keys") and "pretrained_encoder" in h]
assert len(trunks) == 1, trunks

# Tactile keeps its native 240x320 -> 8x10x512 map, cameras stay 4x4x512. Both
# flatten to channel * num_features = 4096, so the bottleneck is unchanged.
assert enc["encoder_tactile"]["SpatialLearnedEmbeddings_0"]["kernel"].shape == (8, 10, 512, 8)
assert enc["encoder_wrist_1"]["SpatialLearnedEmbeddings_0"]["kernel"].shape == (4, 4, 512, 8)
# 512 (2 cams) + 256 (tactile) + 64 (state) + 64 (series) = 896
assert p["network"]["Dense_0"]["kernel"].shape[0] == 896
print("trunk:", trunks[0], "| tactile SLE:", enc["encoder_tactile"]["SpatialLearnedEmbeddings_0"]["kernel"].shape)

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

# The pretrained trunk must actually have been loaded. load_resnet10_params used
# to index by image_keys and silently load nothing once tactile moved out, which
# leaves a randomly initialised trunk that is also frozen. Two different seeds
# giving bit-identical conv_init proves the weights came from the pickle.
other = make_sac_pixel_agent(
    seed=1,
    sample_obs=make_obs(),
    sample_action=np.zeros((ACTION_DIM,), np.float32),
    image_keys=IMAGE_KEYS,
    tactile_keys=TACTILE_KEYS,
    temporal_key=TEMPORAL_KEY,
)
trunk_of = lambda a: a.state.params["modules_actor"]["encoder"][trunks[0]][
    "pretrained_encoder"
]["conv_init"]["kernel"]
assert np.array_equal(trunk_of(agent), trunk_of(other)), "trunk differs across seeds"
print("pretrained trunk loaded (seed-independent)")

# Replay buffer round trip. Tactile is not in image_keys, so it is not a
# pixel_key: obs and next_obs each store a full copy instead of the frame reuse
# the cameras get, which is what makes it safe for a sensor sampled out of step
# with them. Costs 2T x the memory.
space = lambda s, dt, lo, hi: gym.spaces.Box(lo, hi, (1,) + s, dt)
buf = MemoryEfficientReplayBufferDataStore(
    observation_space=gym.spaces.Dict(
        {
            **{k: space(OBS_SHAPES[k], np.uint8, 0, 255) for k in IMAGE_KEYS + TACTILE_KEYS},
            "state": space((STATE_DIM,), np.float32, -np.inf, np.inf),
            TEMPORAL_KEY: space(CHUNK.shape, np.float32, -np.inf, np.inf),
        }
    ),
    action_space=gym.spaces.Box(-1, 1, (ACTION_DIM,), np.float32),
    capacity=16,
    image_keys=IMAGE_KEYS,
)
assert TEMPORAL_KEY not in buf.pixel_keys and TACTILE_KEYS[0] not in buf.pixel_keys
for _ in range(8):
    buf.insert(
        dict(
            observations=base_obs_with_series(),
            next_observations=base_obs_with_series(),
            actions=np.zeros(ACTION_DIM, np.float32),
            rewards=0.0,
            masks=1.0,
            dones=False,
        )
    )
sampled = buf.sample(BATCH)
assert sampled["observations"]["tactile"].shape == (BATCH, 1) + OBS_SHAPES["tactile"]
agent, info = agent.update(frozen_dict.freeze(sampled))
print("buffer critic_loss:", round(float(info["critic"]["critic_loss"]), 4))

print("OK")
