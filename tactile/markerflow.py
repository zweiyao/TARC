"""marker 式触觉传感器 (xense) 的点阵检测与跟踪。

为什么不用稠密光流: marker 间距只有 10px, Farneback 的窗口里只有两个周期,
会大面积锁到错误的点上 —— 合成平移测试中 12px 被混叠成 -6.7px (方向都反了),
真实数据里接触区的流场散乱、不构成合理的形变场。

这里改成显式检测点并匹配到参考帧的点上。本数据实测真实位移只有 1-2.5px, 远低于
半个周期 (5px), 所以直接匹配既无歧义也无漂移。

试过逐帧递推跟踪 (为了突破半周期上限), 但它会累积漂移: 末帧本应回到静止,
直接匹配给 0.17px, 递推跟踪却给 1.57px。只有当真实位移可能超过半个周期时,
才值得用递推并接受漂移 —— 本数据不需要。
"""

import cv2
import numpy as np


def detect(img, thresh=6, amin=3, amax=120):
    """检出 marker 中心。实测全程稳定在 221-223 个, 强接触时也不掉点。"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.GaussianBlur(g, (21, 21), 0)
    m = ((bg.astype(np.int16) - g.astype(np.int16)) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, st, ct = cv2.connectedComponentsWithStats(m, 8)
    keep = [i for i in range(1, n) if amin <= st[i, cv2.CC_STAT_AREA] <= amax]
    return ct[keep].astype(np.float32)


def match(tracked, found, max_dist):
    """把当前检出的点一对一贪心匹配到已跟踪的点上, 未匹配到的保持原位。"""
    if len(found) == 0:
        return tracked
    d = np.linalg.norm(tracked[:, None] - found[None], axis=-1)
    out = tracked.copy()
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    used_t, used_f = set(), set()
    for i, j in order:
        if d[i, j] > max_dist:
            break
        if i in used_t or j in used_f:
            continue
        out[i] = found[j]
        used_t.add(i)
        used_f.add(j)
    return out


class MarkerTracker:
    def __init__(self, ref_img, pitch=10.0):
        self.ref = detect(ref_img)
        self.pitch = pitch
        # 匹配半径取 0.4 个周期: 够覆盖真实位移, 又不会跨到隔壁的点上
        self.max_dist = 0.4 * pitch

    def step(self, img):
        """返回 (点位置 Nx2, 相对参考帧的位移 Nx2)。

        每帧独立匹配到参考帧, 不带跨帧状态, 所以不会漂移。
        """
        pts = match(self.ref, detect(img), self.max_dist)
        return pts, pts - self.ref


def densify(pts, disp, shape, sigma=9.0):
    """把稀疏位移散布成稠密场: 打点后做归一化高斯平滑"""
    acc = np.zeros((*shape, 2), np.float32)
    w = np.zeros(shape, np.float32)
    xi = np.clip(np.round(pts[:, 0]).astype(int), 0, shape[1] - 1)
    yi = np.clip(np.round(pts[:, 1]).astype(int), 0, shape[0] - 1)
    np.add.at(acc, (yi, xi), disp)
    np.add.at(w, (yi, xi), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    w = cv2.GaussianBlur(w, (0, 0), sigma)
    return acc / np.maximum(w[..., None], 1e-6)
