"""Record teleoperated demos for the tarc task.

A fork of record_demos.py for one reason: that file does not touch sys.path, so
`tactile` and `vla` are unimportable from it and building the tarc environment
fails on the first import.

Success is marked by the operator pressing space — KeyboardRewardWrapper turns
that into a non-zero reward, which sets info["succeed"], which is what decides
whether a trajectory is kept (see below). Pressing nothing lets the episode run
to MAX_EPISODE_LENGTH and the trajectory is discarded.

    cd examples/experiments/tarc
    python ../../record_demos_tarc.py --exp_name tarc --successes_needed 20

Run it from the task directory: the output goes to ./demo_data/, which is where
learner.sh looks for it.
"""
import sys
from pathlib import Path

# examples/ for `experiments`, the repo root for `tactile` and `vla`.
_here = Path(__file__).resolve()
sys.path[:0] = [str(_here.parent), str(_here.parents[1])]

import os
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
from absl import app, flags
import time

from experiments.mappings import CONFIG_MAPPING

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("successes_needed", 20, "Number of successful demos to collect.")

def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=False, save_video=False, classifier=False)  # tarc ignores it; reward comes from the keyboard
    
    obs, info = env.reset()
    print("Reset done")
    transitions = []
    success_count = 0
    success_needed = FLAGS.successes_needed
    pbar = tqdm(total=success_needed)
    trajectory = []
    returns = 0
    
    while success_count < success_needed:
        actions = np.zeros(env.action_space.sample().shape) 
        next_obs, rew, done, truncated, info = env.step(actions)
        returns += rew
        if "intervene_action" in info:
            actions = info["intervene_action"]
        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
                infos=info,
            )
        )
        trajectory.append(transition)
        
        pbar.set_description(f"Return: {returns}")

        obs = next_obs
        if done:
            if info["succeed"]:
                for transition in trajectory:
                    transitions.append(copy.deepcopy(transition))
                success_count += 1
                pbar.update(1)
            trajectory = []
            returns = 0
            obs, info = env.reset()
            
    if not os.path.exists("./demo_data"):
        os.makedirs("./demo_data")
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./demo_data/{FLAGS.exp_name}_{success_needed}_demos_{uuid}.pkl"
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_name}")

if __name__ == "__main__":
    app.run(main)