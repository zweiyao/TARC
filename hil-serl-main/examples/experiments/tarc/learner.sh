#!/bin/bash
# Learner on real hardware. DEMO_PATH defaults to the newest pickle
# record_demos_tarc.py wrote; override it to pick a specific one.
PYTHON=${PYTHON:-python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3
DEMO_PATH=${DEMO_PATH:-$(ls -t demo_data/*.pkl 2>/dev/null | head -1)}
if [ -z "$DEMO_PATH" ]; then
    echo "no demos in demo_data/ — run: python ../../record_demos_tarc.py --exp_name tarc" >&2
    exit 1
fi
echo "demos: $DEMO_PATH"
"$PYTHON" ../../train_rlpd_tarc.py "$@" \
    --exp_name=tarc \
    --checkpoint_path=first_run \
    --demo_path="$DEMO_PATH" \
    --learner
