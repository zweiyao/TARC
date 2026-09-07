# π₀.₅ (openpi) inference service

Feeds a π₀.₅ action chunk into the RL network's time-series observation input.

## Why two environments

```
  tarc env                                    openpi env
  Python 3.10, jax 0.4.35                     Python 3.11, jax 0.5.3, torch 2.7.1
  hil-serl + openpi-client       ──ws:8010──> openpi + pi05_droid checkpoint
```

The jax versions are incompatible and hil-serl's 0.4.35 pin is fragile — a
plain `pip install` has already broken the GPU backend once. So the model runs
in its own env and its own process; this env only gets `openpi-client`, whose
dependencies are `dm-tree`, `msgpack`, `numpy<2`, `pillow`, `websockets` — no
jax, no torch.

## Setup (remote box, one time)

openpi pinned at commit `215abfb217dbac7d5f1273282331b9b1866c0479`
(`--depth 1` takes whatever `main` is that day, so record it).

```bash
# 1. clone, outside the repo. Try direct first; if github times out:
#      git config --global http.https://github.com/.proxy http://127.0.0.1:7897
cd ~/holmes && GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 \
  https://github.com/Physical-Intelligence/openpi.git

# 2. its own env (defaults channel has a ToS gate, so conda-forge)
~/miniconda3/bin/conda create -y -n openpi -c conda-forge --override-channels python=3.11
~/miniconda3/envs/openpi/bin/pip install uv

# 3. dependencies. pypi is reachable directly; only the git-sourced lerobot
#    needs the proxy. HF is blocked, hence the mirror.
cd ~/holmes/openpi
GIT_LFS_SKIP_SMUDGE=1 HF_ENDPOINT=https://hf-mirror.com \
UV_PROJECT_ENVIRONMENT=$HOME/miniconda3/envs/openpi \
  ~/miniconda3/envs/openpi/bin/uv sync
#    If uv refuses that env var, drop it and let uv make ~/holmes/openpi/.venv
#    (equally isolated) — then PY= below becomes that venv's python.

# 4. checkpoint, 20 files / 11.6 GiB, straight from GCS.
#    openpi's own maybe_download shells out to gsutil, which isn't installed —
#    hence curl. This path matches its cache layout, so passing --policy.dir
#    as a local absolute path short-circuits the download entirely.
bash /tmp/dl_pi05.sh   # see the script in this repo's history, or the plan file

# 5. client side, in the tarc env. --no-deps is the guard rail: all five
#    dependencies are already present, so pip structurally cannot touch jax.
~/miniconda3/envs/tarc/bin/pip install -c /tmp/tarc_constraints.txt "websockets>=11.0"
~/miniconda3/envs/tarc/bin/pip install --no-deps -e ~/holmes/openpi/packages/openpi-client
```

Verify the tarc env survived: `jax.__version__` must still be `0.4.35` and
`jax.devices()` must still list GPUs.

## Running

```bash
tmux new -s pi05
bash vla/serve_pi05.sh            # loads 11.6 GB, 1-3 min on first start
```

Wait for readiness in another shell:

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8010/healthz   # 200 = up
```

Then, with the server on GPU 0, run the test on a different GPU:

```bash
cd ~/holmes/tarc/hil-serl-main && CUDA_VISIBLE_DEVICES=1 \
  ~/miniconda3/envs/tarc/bin/python -m examples.test.test_temporal_obs_dataflow
```

Both sides preallocate 75% of a GPU by default, so they must not share one.

## Placeholders — not real semantics

This wires up the data flow. The action numbers mean nothing physically.

- **State.** hil-serl's 19-dim state is tcp_pose / tcp_vel / tcp_force /
  tcp_torque / gripper — it has no joint angles, which is precisely what
  pi05_droid expects. `tcp_pose` is dropped into the first six joint slots and
  the seventh is zeroed. The gripper is the only quantity that lines up.
- **Cameras.** `wrist_1` stands in for the exterior view, `wrist_2` for the
  wrist view. hil-serl has no exterior camera.
- **Prompt.** Hardcoded to `"pick up the usb lamp"` in `pi05_client.py`.

## Gotchas

- **A missing server hangs instead of erroring.** `WebsocketClientPolicy`
  retries a refused connection every 5s forever, so `connect()` probes
  `/healthz` first and raises with the commands to fix it.
- **Any inference error kills the server process** — it sends the traceback back
  to the client and then re-raises. Restart it after changing the request
  format; that's why tmux.
- A `RuntimeError` out of `infer()` carries the server's full traceback.
- `pip install -e openpi-client` leaves the tarc env pointing at
  `~/holmes/openpi`. Deleting that clone breaks the import.
