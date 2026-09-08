"""Checks the actor path end to end: franka_env -> wrappers -> policy -> action.

Mirrors what train_rlpd.py's actor does (build the env from CONFIG_MAPPING,
build the agent from the env's spaces, sample one action) and stops there. One
successful inference is the whole pass condition.

The arm, the cameras, the SpaceMouse and the keyboard are all faked, gated on
TARC_FAKE_SOURCES so the real robot path is untouched when it is unset.

    # terminal 1
    python examples/experiments/tarc/fake/franka_server.py
    # terminal 2
    bash vla/serve_pi05.sh
    # terminal 3
    TARC_FAKE_SOURCES=1 python examples/test/test_actor_dataflow.py
"""
import os
import sys
from pathlib import Path

# parents[1] is examples/, which is where `experiments` lives and what
# train_rlpd.py gets for free by sitting there; parents[2] is the repo root, for
# `tactile` and `vla`.
_here = Path(__file__).resolve()
sys.path[:0] = [str(_here.parents[1]), str(_here.parents[2])]

# Before importing anything that reaches franka_env: the camera and SpaceMouse
# drivers are imported at module scope, so the swap has to happen first.
if os.environ.get("TARC_FAKE_SOURCES"):
    from experiments.tarc.fake import install

    install()

import jax
import numpy as np
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.utils.launcher import make_sac_pixel_agent

config = CONFIG_MAPPING["tarc"]()
# fake_env=True cannot be used here even though there is no robot: it returns
# early at franka_env.py:156 and leaves self.cap and self.terminate undefined,
# so reset() and step() both die. The fakes above are what make fake_env=False
# viable. classifier=False skips the checkpoint load and the ResNet-10 download.
env = config.get_environment(fake_env=False, save_video=False, classifier=False)
env = RecordEpisodeStatistics(env)

obs, _ = env.reset()
print("obs keys:", {k: (v.shape, v.dtype) for k, v in sorted(obs.items())})

# The action chunk is the key SERLObsWrapper used to drop on the floor: it is
# neither "state" nor an image, so before this change it vanished between the
# env and the policy without an error anywhere.
assert config.action_chunk_key in obs, sorted(obs)
assert obs["state"].shape == (1, 19), obs["state"].shape
assert obs["tactile"].shape == (1, 240, 320, 3), obs["tactile"].shape

agent = make_sac_pixel_agent(
    seed=0,
    sample_obs=env.observation_space.sample(),
    sample_action=env.action_space.sample(),
    image_keys=config.image_keys,
    tactile_keys=config.tactile_keys,
    action_chunk_key=config.action_chunk_key,
    encoder_type=config.encoder_type,
    discount=config.discount,
)

action = agent.sample_actions(
    observations=jax.device_put(obs), seed=jax.random.PRNGKey(0), argmax=False
)
action = np.asarray(jax.device_get(action))
assert action.shape == env.action_space.shape, (action.shape, env.action_space.shape)
assert np.isfinite(action).all() and np.abs(action).max() <= 1.0
print("action:", action.round(3))

# One step, mostly to prove the /pose -> /getstate loop in the fake server is
# closed: a stateless stub would leave the pose unchanged and hide any mistake
# in how actions are sent.
before = env.unwrapped.currpos.copy()
obs, reward, done, truncated, info = env.step(action)
moved = float(np.abs(env.unwrapped.currpos - before).max())
print(f"step ok, reward={reward}, pose moved {moved:.4f}")
assert moved > 0, "pose did not change — is the fake server stateful?"

env.close()
print("OK")
