"""Synthetic demo transitions, so the learner's demo buffer is not empty.

record_demos.py cannot run without hardware: it drives the arm from the
SpaceMouse, and the fake one returns zeros, which sits below the 0.001
intervention threshold at wrappers.py:230 — so info["intervene_action"] never
appears, info["succeed"] never becomes true, and the script loops forever.

This rolls out the fake environment with random actions instead of fabricating
dicts by hand. Rolling out gets four invariants right for free, and every one of
them corrupts training silently when a hand-written version gets it wrong:

  - transitions in trajectory order (the buffer reconstructs obs images from the
    previous row, memory_efficient_replay_buffer.py:151-158)
  - masks == 1 - dones (sac.py:181 multiplies the bootstrap by masks)
  - every observation key present in both obs and next_obs, with the leading
    obs_horizon axis
  - shapes that match env.observation_space exactly

    TARC_FAKE_SOURCES=1 python gen_demos.py --episodes 5 --steps 20

Writes demo_data/tarc_{n}_demos_{timestamp}.pkl, the same layout and naming
record_demos.py:62-65 produces, so --demo_path takes it unchanged.
"""
import argparse
import datetime
import os
import pickle as pkl
import sys
from pathlib import Path

_here = Path(__file__).resolve()
sys.path[:0] = [str(_here.parents[2]), str(_here.parents[3])]

if os.environ.get("TARC_FAKE_SOURCES"):
    from experiments.tarc.fake import install

    install()

import numpy as np

from experiments.mappings import CONFIG_MAPPING


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20, help="steps per episode")
    ap.add_argument("--out_dir", default="demo_data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    config = CONFIG_MAPPING["tarc"]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)
    env.action_space.seed(a.seed)

    transitions = []
    for ep in range(a.episodes):
        obs, _ = env.reset()
        for t in range(a.steps):
            action = env.action_space.sample()
            next_obs, _, _, _, info = env.step(action)
            # The last step of each trajectory is marked a success. Real demos
            # look like this too: record_demos.py:52-60 keeps a trajectory only
            # when info["succeed"] is set, so its final step always carries the
            # reward. An all-zero demo buffer teaches nothing.
            last = t == a.steps - 1
            transitions.append(
                dict(
                    observations=obs,
                    actions=action,
                    next_observations=next_obs,
                    rewards=1.0 if last else 0.0,
                    masks=0.0 if last else 1.0,
                    dones=last,
                    infos=info,
                )
            )
            obs = next_obs
        print(f"  episode {ep + 1}/{a.episodes}: {len(transitions)} transitions so far")

    env.close()

    os.makedirs(a.out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(a.out_dir, f"tarc_{a.episodes}_demos_{stamp}.pkl")
    with open(path, "wb") as f:
        pkl.dump(transitions, f)

    size = os.path.getsize(path) / 1e6
    print(f"\n{len(transitions)} transitions -> {path}  ({size:.1f} MB)")
    print("keys:", {k: np.asarray(v).shape for k, v in transitions[0]["observations"].items()})


if __name__ == "__main__":
    main()
