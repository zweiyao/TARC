#!/bin/bash
# Learner against the fake robot. DEMO_PATH defaults to the newest pickle
# gen_demos.py wrote; override it to point at real demos.
export TARC_FAKE_SOURCES=1
# The original run_*.sh assume `conda activate tarc` first. PYTHON lets a
# caller point at the interpreter instead, which non-interactive shells need.
PYTHON=${PYTHON:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
DEMO_PATH=${DEMO_PATH:-$(ls -t demo_data/*.pkl 2>/dev/null | head -1)}
if [ -z "$DEMO_PATH" ]; then
    echo "no demos in demo_data/ — run: TARC_FAKE_SOURCES=1 python gen_demos.py" >&2
    exit 1
fi
echo "demos: $DEMO_PATH"
# --debug turns off wandb, which needs an API key this box does not have.
"$PYTHON" ../../train_rlpd_tarc_fake.py "$@" \
    --exp_name=tarc \
    --checkpoint_path=first_run_fake \
    --demo_path="$DEMO_PATH" \
    --debug \
    --learner
