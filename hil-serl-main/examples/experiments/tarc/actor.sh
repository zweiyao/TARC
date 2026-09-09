#!/bin/bash
# Actor against the fake robot. run_actor.sh is the real-hardware one; this
# differs only in TARC_FAKE_SOURCES and the entry point.
export TARC_FAKE_SOURCES=1
# The original run_*.sh assume `conda activate tarc` first. PYTHON lets a
# caller point at the interpreter instead, which non-interactive shells need.
PYTHON=${PYTHON:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
# --debug turns off wandb, which needs an API key this box does not have.
"$PYTHON" ../../train_rlpd_tarc.py "$@" \
    --exp_name=tarc \
    --checkpoint_path=first_run \
    --debug \
    --actor
