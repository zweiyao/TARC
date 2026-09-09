#!/bin/bash
# Actor on real hardware. franka_server, tactile/capture.py and vla/serve_pi05.sh
# must already be running; see actor_fake.sh for the version that stands them in.
#
# The upstream run_*.sh assume `conda activate tarc` first. PYTHON lets a caller
# name the interpreter instead, which non-interactive shells need.
PYTHON=${PYTHON:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
"$PYTHON" ../../train_rlpd_tarc.py "$@" \
    --exp_name=tarc \
    --checkpoint_path=first_run \
    --actor
