"""打印各网络参数量。改完网络结构后跑这个，确认改动真的生效了。

    python -m examples.test.count_params        # 从仓库根目录运行
"""
import numpy as np
import jax

from examples.test.smoke_learner import fake_obs, ACTION_DIM, IMAGE_KEYS
from serl_launcher.utils.launcher import make_sac_pixel_agent


def count(tree):
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))


agent = make_sac_pixel_agent(
    seed=0,
    sample_obs=fake_obs(),
    sample_action=np.zeros((ACTION_DIM,), np.float32),
    image_keys=IMAGE_KEYS,
)
p = agent.state.params

# 注意：顶层键是 modules_critic / modules_actor / modules_temperature，
# 不是 critic / actor / temperature。这是 ModuleDict 加的前缀。
critic = p["modules_critic"]
actor = p["modules_actor"]

print(f"critic 总计        {count(critic):>12,}")
print(f"  network (MLP)    {count(critic['network']):>12,}   ← 改 hidden_dims 影响这里")
print(f"  Dense_0 (输出层) {count(critic['Dense_0']):>12,}")
print(f"actor  总计        {count(actor):>12,}")
print(f"  encoder (ResNet) {count(actor['encoder']):>12,}   ← 冻结的预训练权重")
print(f"  network (MLP)    {count(actor['network']):>12,}")
print(f"全部参数           {count(p):>12,}")
