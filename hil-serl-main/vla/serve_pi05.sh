#!/usr/bin/env bash
# Serve pi0.5 (pi05_droid). Runs in the openpi env, NOT tarc — see vla/README.md.
# Every path can be overridden by an environment variable.
set -euo pipefail
: "${OPENPI_DIR:=$HOME/holmes/openpi}"
: "${CKPT_DIR:=$HOME/.cache/openpi/openpi-assets/checkpoints/pi05_droid}"
: "${PY:=$HOME/miniconda3/envs/openpi/bin/python}"
# 8000 is taken by another service on the shared box.
: "${PORT:=8010}"
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
[ -d "$CKPT_DIR/params" ] || { echo "missing checkpoint: $CKPT_DIR (see vla/README.md)"; exit 1; }
cd "$OPENPI_DIR"
# tyro binds options to the directly preceding subcommand, so --port must come
# before policy:checkpoint.
exec "$PY" scripts/serve_policy.py --port "$PORT" policy:checkpoint \
  --policy.config=pi05_droid --policy.dir="$CKPT_DIR"
