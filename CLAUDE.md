# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TARC is a fork of hil-serl. The goal, from the README: **触觉在精细阶段介入，高频修正 vla** — tactile sensing intervenes during fine manipulation to make high-frequency corrections to a VLA's plan. Concretely, a π₀.₅ action chunk and a tactile optical-flow heatmap are fed alongside the two wrist cameras into an RLPD agent.

Upstream hil-serl code is left as close to original as possible. TARC-specific work lives in `examples/experiments/tarc/`, `tactile/`, `vla/`, and the `*_tarc*.py` entry points; the other tasks under `examples/experiments/` are upstream and should not be "improved" in passing.

## Development runs on a remote box, not locally

Edit locally at `/Users/zweiyao/work/tarc`; **everything executes on the remote box** — there is no GPU and no hardware on the Mac.

- `ssh zhangweiyao@10.176.53.118`, repo at `~/holmes/tarc`
- Conda is **not** on PATH for non-interactive ssh. Use absolute paths: `~/miniconda3/envs/tarc/bin/python`
- Every ssh prints `Warning: remote port forwarding failed for listen port 7897`. Harmless — filter it out.
- 8× RTX 4090, **shared with other people**. Check `nvidia-smi` and pick a free GPU; always set `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

After editing anything, run the sync yourself in the same turn — do not end a turn telling the user to run it:

```bash
rsync -az --delete --exclude .git --exclude '.DS_Store' --exclude '__pycache__' \
  --exclude 'demo_data' --exclude 'videos' --exclude 'first_run*' \
  --exclude 'wandb' --exclude '_scratch' \
  /Users/zweiyao/work/tarc/ zhangweiyao@10.176.53.118:holmes/tarc/
```

Every exclude is load-bearing: `--delete` runs at the repo root, and dropping one wipes recorded demos, 600 MB+ checkpoint dirs, or the debug scripts parked in `_scratch/`. Dry-run with `-n` before any sync whose exclude list changed.

**`serl_launcher` and `serl_robot_infra` are editable installs whose absolute paths are baked into site-packages.** Moving or renaming the repo directory breaks imports until both are reinstalled. `serl_robot_infra` must use `pip install -e . --no-deps --config-settings editable_mode=compat` — plain `find_packages()` misses `franka_env`, which is a namespace package (so its `__file__` is `None`; check `__path__` instead).

Version pins are fragile — jax 0.4.35 + numpy<2 in the `tarc` env. A plain `pip install` has already broken the GPU backend once. π₀.₅ therefore runs in a **separate conda env and process** (`openpi`, Python 3.11, jax 0.5.3), reached over a websocket; this env only carries `openpi-client`.

## Commands

All from the repo root on the box. `PY=~/miniconda3/envs/tarc/bin/python`.

**Tests needing no services:**
```bash
$PY examples/test/test_live_vla.py         # VLA cadence/threading, stubbed client
$PY examples/test/test_live_tactile.py     # tactile over real shared memory
$PY -m examples.test.smoke_learner         # agent construction + one update
$PY -m examples.test.count_params          # parameter counts after a network change
```

**Tests needing services up first:**
```bash
$PY examples/test/test_actor_dataflow.py        # needs fake franka + pi0.5, TARC_FAKE_SOURCES=1
$PY -m examples.test.test_temporal_obs_dataflow # needs pi0.5
```

`test_temporal_obs_dataflow.py` advertises two invocation styles (`python -m` and by file path). **Verify both** — a `sys.path` bug once passed review because only `python -m` was tried.

**Services:**
```bash
bash vla/serve_pi05.sh                                  # π₀.₅ on :8010, own conda env
$PY examples/experiments/tarc/fake/franka_server.py     # fake arm on :5010
$PY tactile/capture.py                                  # owns /dev/video0 -> shared memory
```
π₀.₅'s first inference compiles for ~14 s (81 ms steady state). That is not a hang.

**Training** — actor and learner are two separate processes, launched from `examples/experiments/tarc/`:
```bash
PYTHON=$PY bash learner_fake.sh    # then, in another shell:
PYTHON=$PY bash actor_fake.sh
```
`actor.sh` / `learner.sh` are the real-hardware pair. The `_fake` pair sets `TARC_FAKE_SOURCES=1` and `--debug` (wandb off). `gen_demos.py` produces synthetic demos so the learner's demo buffer is not empty.

## Architecture

### Actor–learner split

One file, `train_rlpd_tarc.py`, run twice with `--actor` or `--learner`, communicating over agentlace (port 5588 data, 5589 param broadcast; `make_trainer_config` in `serl_launcher/serl_launcher/utils/launcher.py`). The actor steps the env and ships transitions; the learner updates and broadcasts parameters.

`main()` passes **`fake_env=FLAGS.learner`**. The learner builds the env only to read its `observation_space`, and gets spaces-only stand-ins for tactile and VLA — it never touches the arm, the tactile camera, or π₀.₅.

### Three observation modalities

| | enters as | produced by |
|---|---|---|
| wrist_1 / wrist_2 | `obs["images"]` | `franka_env.get_im()` |
| tactile | `obs["images"]["tactile"]` | `tactile/live.py`, background thread |
| action_chunk | its own top-level key | `vla/live.py`, background thread |

`TactileVLAWrapper` (`experiments/tarc/wrapper.py`) injects both. The chunk is a top-level key rather than an image because everything that iterates `obs["images"]` would otherwise trip over it — this is why `SERLObsWrapper` has an `extra_keys` pass-through.

### Wrapper chain, where order is load-bearing

```
TarcEnv → GripperCloseEnv → [SpacemouseIntervention] → RelativeFrame
  → Quat2EulerWrapper → TactileVLAWrapper → SERLObsWrapper
  → ChunkingWrapper → KeyboardRewardWrapper
```

Two constraints, both of which fail *silently* if broken:
- **after `Quat2EulerWrapper`** — `tcp_pose` is 6-dim euler there, giving the 19-dim state π₀.₅ is handed. Before it, it is a 7-dim quaternion and the totals disagree (20 vs 19) without raising.
- **before `SERLObsWrapper`** — that is what lifts `obs["images"]` to the top level.

### Encoder: three branches, one shared frozen trunk

`EncodingWrapper` (`serl_launcher/serl_launcher/common/encoding.py`) has an image branch, a tactile branch and an action-chunk branch. Tactile goes through the **same** frozen ImageNet ResNet-10 trunk as the cameras — sharing works by Flax instance identity, so per-head settings (`normalize`, `do_resize`) must be passed as **call-time kwargs**, not dataclass fields, or the trunk splits into separate copies.

Three key lists that are easy to conflate:

- `image_keys` — encoder image branch **and** the crop augmentation
- `tactile_keys` — encoder tactile branch; deliberately *not* in `image_keys`, so it skips the crop augmentation
- `buffer_pixel_keys` = `image_keys + tactile_keys` — replay-buffer storage and frame reuse only, orthogonal to the encoder

`make_sac_pixel_agent_hybrid_single_arm` / `_dual_arm` (`launcher.py:103`, `:153`) **do not accept `tactile_keys` or `action_chunk_key` at all**. TARC uses `setup_mode = "single-arm-fixed-gripper"` so it reaches `make_sac_pixel_agent`; changing `setup_mode` would silently drop tactile and VLA and build a 576-wide actor. `train_rlpd_tarc.py` raises `NotImplementedError` on those branches rather than letting it pass.

### Control-loop rate limiting — the thing that bites

`franka_env.py:231` is the only rate limit, and its **placement** is the whole story: the sleep runs *before* `_update_currpos()` and `_get_obs()`, and `dt` only spans the command send. So observation cost is **net-added** to the period, not absorbed by it. Anything put in an `ObservationWrapper` costs control frequency one-for-one.

That is why both tactile and VLA run on background threads. Measured: synchronous π₀.₅ gave 190 ms (5.26 Hz); with `vla/live.py` it is 106 ms (9.41 Hz). The VLA fires once every `VLA_EVERY = 5` steps (≈2 Hz) and the policy reads the latest published chunk, so the same chunk appears in five consecutive observations.

Both live sources follow the same contract: block in `__init__` until a first valid value exists (the wrapper probes them to size the observation space), store background-thread exceptions and re-raise on the caller's thread, and raise on staleness rather than serving a frozen value — a silently frozen observation is far worse than a crash.

### Fake hardware path

`TARC_FAKE_SOURCES=1` plus `experiments/tarc/fake/install()`, which **must run before `franka_env` is imported** — the camera and SpaceMouse drivers are pulled in at module scope, so patching after the env class exists is too late. The fake arm is a separate HTTP process (`fake/franka_server.py`, port 5010). **π₀.₅ is not faked** and must be running either way.

### Reward

`KeyboardRewardWrapper` replaces the trained classifier: an operator presses space for success. Hence `classifier=False` throughout the tarc path and no `classifier_ckpt/` to collect data for.

## Conventions

- Comments explain **why**, and cite `file.py:line` for the code that constrains the decision. Match that density and style; it is what makes the silent-failure traps above discoverable.
- Commit messages and PR descriptions are written in Chinese.
- Prefer measuring over estimating — several decisions here (VLA cadence, flow resolution, buffer sizing) are justified by numbers recorded in `tactile/README.md`, `vla/README.md` and the module docstrings.
