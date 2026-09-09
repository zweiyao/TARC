"""
learner 侧冒烟测试：不接机器人、不需要 demo 数据，
直接用假 batch 把 agent.update() 跑通。

用途：改完 learner 侧代码后，先跑这个确认没把形状/网络结构改坏，
再去跑真正的训练。几秒钟出结果。

    python -m examples.test.smoke_learner        # 从仓库根目录运行
"""
import numpy as np
import jax, jax.numpy as jnp
from flax.core import frozen_dict

from serl_launcher.utils.launcher import make_sac_pixel_agent

# 下面每个数字都是从代码里读出来的，不是猜的。改任务时按同样方法重新推导：
#
# IMAGE_KEYS   config.py:93        image_keys = ["wrist_1", "wrist_2"]
# 128x128x3    franka_env.py:149   Box(0, 255, shape=(128,128,3))
# HORIZON=1    config.py:114       ChunkingWrapper(env, obs_horizon=1, ...)
# PROPRIO_DIM  config.py:95        proprio_keys = [tcp_pose, tcp_vel, tcp_force,
#                                                  tcp_torque, gripper_pose]
#              franka_env.py:136   原始维度 7 + 6 + 3 + 3 + 1 = 20
#              wrappers.py:112     Quat2EulerWrapper 把 tcp_pose 7 -> 6，故 19
# ACTION_DIM   franka_env.py:130   原始 action 7 维
#              wrappers.py:189     GripperCloseEnv 砍成前 6 维
IMAGE_KEYS = ["wrist_1", "wrist_2"]
PROPRIO_DIM = 19
ACTION_DIM = 6
BATCH = 8
HORIZON = 1


# 用随机数而不是全零：全零经过第一层 Dense 后只剩 bias（初始为 0），
# 会把维度、权重的差异全部抹平，导致测试对错误不敏感。
rng = np.random.default_rng(0)


def fake_obs(batch=None):
    """构造和真环境同形状的观测。batch=None 时是单帧（给 create 用）。"""
    shape = lambda *s: (batch, HORIZON, *s) if batch else (HORIZON, *s)
    obs = {k: rng.integers(0, 256, shape(128, 128, 3), np.uint8) for k in IMAGE_KEYS}
    obs["state"] = rng.standard_normal(shape(PROPRIO_DIM), np.float32)
    return obs


def fake_batch():
    # 缓冲区返回的是 FrozenDict（memory_efficient_replay_buffer.py:167），
    # data_augmentation_fn 依赖 .copy(add_or_replace=...)，普通 dict 会报错
    return frozen_dict.freeze({
        "observations": fake_obs(BATCH),
        "next_observations": fake_obs(BATCH),
        "actions": rng.uniform(-1, 1, (BATCH, ACTION_DIM)).astype(np.float32),
        "rewards": rng.uniform(0, 1, (BATCH,)).astype(np.float32),
        "masks": rng.integers(0, 2, (BATCH,)).astype(np.float32),
    })


def show(tag, info):
    """info 是嵌套 dict（critic/actor/temperature 各一层），拍平后只打印标量。"""
    flat = {}
    def walk(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(v, f"{prefix}{k}/")
            elif np.ndim(v) == 0:
                flat[f"{prefix}{k}"] = round(float(v), 4)
    walk(info)
    print(f"{tag}:", flat)


def main():
    agent = make_sac_pixel_agent(
        seed=0,
        sample_obs=fake_obs(),
        sample_action=np.zeros((ACTION_DIM,), np.float32),
        image_keys=IMAGE_KEYS,
        encoder_type="resnet-pretrained",
    )
    print("agent 构造成功")

    batch = fake_batch()
    agent, info = agent.update(batch, networks_to_update=frozenset({"critic"}))
    show("critic 更新", info)

    agent, info = agent.update(
        batch, networks_to_update=frozenset({"critic", "actor", "temperature"})
    )
    show("全网络更新", info)
    print("\nOK")


if __name__ == "__main__":
    main()
