"""Data-flow test for the extra 50x7 time-series observation input.

Feeds a synthetic image + synthetic 19-dim proprio state + an all-ones 50x7
matrix through the policy and checks that an action comes out AND that the new
input actually took part in the computation.

No robot, no server, no demo data. Run from the repo root:

    python examples/test/test_temporal_obs_dataflow.py
"""
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import frozen_dict

from serl_launcher.utils.launcher import make_sac_pixel_agent

# Shapes traced through the wrapper chain; see examples/test/smoke_learner.py
IMAGE_KEYS = ["wrist_1", "wrist_2"]
IMG_SHAPE = (128, 128, 3)
STATE_DIM = 19
SERIES_T, SERIES_C = 50, 7
ACTION_DIM = 6
TEMPORAL_KEY = "series"
LATENT_DIM = 64

# 256 per image + 64 proprio; the temporal branch adds another 64
ENC_DIM_BASE = 256 * len(IMAGE_KEYS) + 64
ENC_DIM_TEMPORAL = ENC_DIM_BASE + LATENT_DIM

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def make_obs(series_fill=1.0, batch=None, seed=0, with_series=True):
    """Same shapes the wrappers would produce, built by hand."""
    rng = np.random.RandomState(seed)
    lead = (1,) if batch is None else (batch, 1)
    obs = {
        k: rng.randint(0, 256, lead + IMG_SHAPE).astype(np.uint8) for k in IMAGE_KEYS
    }
    obs["state"] = rng.standard_normal(lead + (STATE_DIM,)).astype(np.float32)
    if with_series:
        if series_fill is None:
            obs[TEMPORAL_KEY] = rng.standard_normal(
                lead + (SERIES_T, SERIES_C)
            ).astype(np.float32)
        else:
            obs[TEMPORAL_KEY] = np.full(
                lead + (SERIES_T, SERIES_C), series_fill, np.float32
            )
    return obs


def build(temporal=True):
    kw = dict(temporal_key=TEMPORAL_KEY) if temporal else {}
    return make_sac_pixel_agent(
        seed=0,
        sample_obs=make_obs(with_series=temporal),
        sample_action=np.zeros((ACTION_DIM,), np.float32),
        image_keys=IMAGE_KEYS,
        **kw,
    )


def norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return float(jnp.sqrt(sum(jnp.sum(x**2) for x in leaves))) if leaves else 0.0


def main():
    key = jax.random.PRNGKey(0)
    agent = build(temporal=True)
    p = agent.state.params
    enc = p["modules_actor"]["encoder"]

    print("\n[1] 结构：确认是时序卷积，不是 flatten")
    check("temporal_encoder 在参数树里", "temporal_encoder" in enc)
    kshape = enc["temporal_encoder"]["conv_0"]["kernel"].shape
    check(
        "conv_0 kernel 跨越时间窗口",
        kshape == (5, SERIES_C, 64),
        f"{kshape} == (kernel=5, C=7, feat=64)",
    )
    flat_dim = SERIES_T * SERIES_C
    has_flat = any(
        flat_dim in tuple(v.shape) for v in jax.tree_util.tree_leaves(p)
    )
    check(
        f"参数树中不存在 {flat_dim} 维（未被偷换成 flatten+Dense）", not has_flat
    )

    print("\n[2] 接线：actor 和 critic 都吃到了新输入")
    a_shape = p["modules_actor"]["network"]["Dense_0"]["kernel"].shape
    c_shape = p["modules_critic"]["network"]["VmapMLP_0"]["Dense_0"]["kernel"].shape
    check("actor MLP 首层已加宽", a_shape == (ENC_DIM_TEMPORAL, 256), f"{a_shape}")
    check(
        "critic MLP 首层已加宽",
        c_shape == (2, ENC_DIM_TEMPORAL + ACTION_DIM, 256),
        f"{c_shape}",
    )
    check("encoder 参数仍只存一份", "encoder" not in p["modules_critic"])

    print("\n[3] 形状：推理路径与训练路径")
    obs = make_obs(1.0)
    a = agent.sample_actions(obs, seed=key, argmax=True)
    check("推理输出动作形状", a.shape == (ACTION_DIM,), f"{a.shape}")
    check("动作有限", bool(np.all(np.isfinite(a))))
    check("动作被 tanh 压在 [-1,1]", bool(np.all(np.abs(a) <= 1.0 + 1e-6)))

    obs_b = make_obs(1.0, batch=4)
    act_b = jnp.zeros((4, ACTION_DIM), jnp.float32)
    m_b = agent.forward_policy(obs_b, rng=key, train=False).mode()
    q_b = agent.forward_critic(obs_b, act_b, rng=key, train=False)
    check("批量 policy 输出", m_b.shape == (4, ACTION_DIM), f"{m_b.shape}")
    check("批量 critic 输出", q_b.shape == (2, 4), f"{q_b.shape}")

    print("\n[4] 影响性：新输入必须真的改变输出")
    mode = lambda o: np.asarray(
        agent.forward_policy(o, rng=key, train=False).mode()
    )
    m_ones = mode(make_obs(1.0))
    check("同一输入两次完全一致（对照组）", np.array_equal(m_ones, mode(make_obs(1.0))))
    d_zero = float(np.max(np.abs(m_ones - mode(make_obs(0.0)))))
    d_rand = float(np.max(np.abs(m_ones - mode(make_obs(None)))))
    check("series 全1 vs 全0 输出不同", d_zero > 1e-4, f"最大差 {d_zero:.4f}")
    check("series 全1 vs 随机 输出不同", d_rand > 1e-4, f"最大差 {d_rand:.4f}")

    print("\n[5] 梯度：计算图连通 + 回归保护")
    base = make_obs(1.0)

    def act_sum(series):
        o = dict(base)
        o[TEMPORAL_KEY] = series
        return jnp.sum(agent.forward_policy(o, rng=key, train=False).mode())

    g = jax.grad(act_sum)(jnp.ones((1, SERIES_T, SERIES_C), jnp.float32))
    check("对 series 输入的梯度形状", g.shape == (1, SERIES_T, SERIES_C), f"{g.shape}")
    check("梯度有限且非零", bool(np.all(np.isfinite(g))) and norm(g) > 1e-6,
          f"范数 {norm(g):.4f}")

    def actor_sum(params):
        return jnp.sum(
            agent.forward_policy(base, rng=key, train=False, grad_params=params).mode()
        )

    gp = jax.grad(actor_sum)(p)["modules_actor"]["encoder"]
    n_temp = norm(gp["temporal_encoder"])
    n_img = norm(gp[f"encoder_{IMAGE_KEYS[0]}"])
    check("actor 梯度能训 temporal_encoder", n_temp > 0, f"范数 {n_temp:.4f}")
    check(
        "actor 梯度未漏进图像 encoder（回归保护）", n_img == 0.0, f"范数 {n_img}"
    )

    def critic_sum(params):
        return jnp.sum(
            agent.forward_critic(obs_b, act_b, rng=key, train=False, grad_params=params)
        )

    gc = jax.grad(critic_sum)(p)["modules_actor"]["encoder"]["temporal_encoder"]
    check("critic 梯度也能训 temporal_encoder", norm(gc) > 0, f"范数 {norm(gc):.4f}")

    print("\n[6] 端到端：跑一步真实 update()")
    batch = frozen_dict.freeze(
        {
            "observations": make_obs(1.0, batch=4, seed=1),
            "next_observations": make_obs(1.0, batch=4, seed=2),
            "actions": np.zeros((4, ACTION_DIM), np.float32),
            "rewards": np.zeros((4,), np.float32),
            "masks": np.ones((4,), np.float32),
        }
    )
    new_agent, info = agent.update(batch)
    losses = {
        f"{net}_loss": float(info[net][f"{net}_loss"])
        for net in ("critic", "actor", "temperature")
    }
    check(
        "update 后 loss 有限",
        all(np.isfinite(v) for v in losses.values()),
        str({k: round(v, 4) for k, v in losses.items()}),
    )
    before = enc["temporal_encoder"]["conv_0"]["kernel"]
    after = new_agent.state.params["modules_actor"]["encoder"]["temporal_encoder"][
        "conv_0"
    ]["kernel"]
    delta = float(jnp.max(jnp.abs(after - before)))
    check("temporal_encoder 参数确实被更新", delta > 0, f"最大变化 {delta:.2e}")

    print("\n[7] 向后兼容：不传 temporal_key 时行为不变")
    agent2 = build(temporal=False)
    p2 = agent2.state.params
    check(
        "参数树无 temporal_encoder",
        "temporal_encoder" not in p2["modules_actor"]["encoder"],
    )
    a2_shape = p2["modules_actor"]["network"]["Dense_0"]["kernel"].shape
    check("actor MLP 首层维持原宽度", a2_shape == (ENC_DIM_BASE, 256), f"{a2_shape}")
    a2 = agent2.sample_actions(make_obs(with_series=False), seed=key, argmax=True)
    check("原路径仍能出动作", a2.shape == (ACTION_DIM,), f"{a2.shape}")

    failed = [n for n, ok in _results if not ok]
    print(f"\n{'=' * 60}")
    print(f"通过 {len(_results) - len(failed)}/{len(_results)}")
    if failed:
        print("失败项:")
        for n in failed:
            print(f"  - {n}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
