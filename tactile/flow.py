"""相机式触觉传感器: 稠密光流位移场 -> 热力图

凝胶表面是随机 RGB 散斑, 等价于 DIC 里的散斑图案, 因此稠密光流可解。

核心约束: 光流始终相对固定的无接触参考帧计算 flow(ref -> cur),
不是相邻帧 flow(prev -> cur)。触觉信号表达的是形变状态而非速度,
静止按压时形变持续存在, 相邻帧光流会归零。
"""

import cv2
import numpy as np

# 实测局部纹理对比度 (9x9 std): R=3.17 G=2.55 B=2.08 灰度=2.06
# R 通道比灰度强 50%, 所以不转灰度而是直接取 R (BGR 里索引 2)
RED = 2


class TactileFlow:
    def __init__(self, size=(960, 540), channel=RED, ema=0.5, flip_div=False,
                 winsize=21, clip_limit=4.0):
        self.size = size
        self.channel = channel
        self.ema = ema
        self.flip_div = flip_div
        self.winsize = winsize
        # 绝对对比度只有约 3/255, 不做局部对比度归一化光流信噪比不够
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        self.ref = None
        self.ref_bgr = None
        self.flow = None

    def _prep(self, frame):
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return small, self.clahe.apply(small[:, :, self.channel])

    def set_reference(self, frames):
        """N 帧平均建立基线, 抑制传感器读出噪声。必须在无接触状态调用。"""
        acc = None
        for f in frames:
            small, _ = self._prep(f)
            acc = small.astype(np.float32) if acc is None else acc + small
        mean = (acc / len(frames)).astype(np.uint8)
        self.ref_bgr = mean
        self.ref = self.clahe.apply(mean[:, :, self.channel])
        self.flow = None

    def process(self, frame):
        small, cur = self._prep(frame)
        flow = cv2.calcOpticalFlowFarneback(
            self.ref, cur, None,
            pyr_scale=0.5, levels=3, winsize=self.winsize,
            iterations=3, poly_n=7, poly_sigma=1.5, flags=0,
        )
        self.flow = flow if self.flow is None else self.ema * flow + (1 - self.ema) * self.flow

        u, v = self.flow[..., 0], self.flow[..., 1]
        mag = cv2.GaussianBlur(np.sqrt(u * u + v * v), (9, 9), 0)
        # 散度: 压下去时凝胶被撑开, 散斑径向外散; 横向抹动散度接近 0。
        # 先平滑位移场再求导 —— 求导会放大噪声, 顺序反了散度就全是噪声。
        # ksize=3 配 scale=1/8 是归一化的中心差分 (核对斜率 1 的斜坡响应为 8),
        # 因此 div 的量纲是无量纲应变, 而不是任意单位。
        us = cv2.GaussianBlur(u, (9, 9), 0)
        vs = cv2.GaussianBlur(v, (9, 9), 0)
        div = (cv2.Sobel(us, cv2.CV_32F, 1, 0, ksize=3, scale=0.125)
               + cv2.Sobel(vs, cv2.CV_32F, 0, 1, ksize=3, scale=0.125))
        if self.flip_div:
            div = -div
        diff = cv2.absdiff(small, self.ref_bgr).max(axis=2)
        return {"small": small, "flow": self.flow, "mag": mag, "div": div, "diff": diff}


def _diverging_lut():
    """cv2 没有内置发散色表, 手工构造 蓝(负)-白(0)-红(正)"""
    x = np.linspace(-1.0, 1.0, 256)
    lut = np.zeros((256, 1, 3), np.uint8)
    lut[:, 0, 0] = np.clip(np.where(x < 0, 255.0, 255.0 * (1 - x)), 0, 255)  # B
    lut[:, 0, 1] = np.clip(255.0 * (1 - np.abs(x)), 0, 255)                  # G
    lut[:, 0, 2] = np.clip(np.where(x > 0, 255.0, 255.0 * (1 + x)), 0, 255)  # R
    return lut


_DIV_LUT = _diverging_lut()


def _colorize(field, vmax, cmap):
    """固定色标上下限, 不用自动归一化, 否则不同时刻的图不可比"""
    norm = np.clip(field / vmax, 0.0, 1.0)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)


def _colorize_signed(field, vmax):
    norm = np.clip(field / vmax * 0.5 + 0.5, 0.0, 1.0)
    gray = cv2.cvtColor((norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return cv2.LUT(gray, _DIV_LUT)


def _draw_quiver(img, flow, n=24, scale=6.0, thresh=0.2):
    h, w = flow.shape[:2]
    step = max(h // n, 1)
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            u, v = flow[y, x]
            if u * u + v * v < thresh * thresh:
                continue  # 静止处不画, 否则噪声箭头糊满屏
            cv2.arrowedLine(img, (x, y), (int(x + u * scale), int(y + v * scale)),
                            (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)


def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render(fields, vmax=3.0, div_vmax=1.0, diff_vmax=40.0, stats=""):
    """2x2 面板: 原始 | 差分 | 位移大小+箭头 | 散度"""
    mag, div = fields["mag"], fields["div"]

    raw = _label(fields["small"].copy(), "raw")
    diff = _label(_colorize(fields["diff"].astype(np.float32), diff_vmax,
                            cv2.COLORMAP_INFERNO), "|cur - ref|")
    mag_img = _colorize(mag, vmax, cv2.COLORMAP_TURBO)
    _draw_quiver(mag_img, fields["flow"])
    mag_img = _label(mag_img, f"|d| shear  vmax={vmax}px  mean={mag.mean():.2f} p99={np.percentile(mag, 99):.2f}")
    div_img = _label(_colorize_signed(div, div_vmax), f"div(d) press  vmax={div_vmax}")

    panel = np.vstack([np.hstack([raw, diff]), np.hstack([mag_img, div_img])])
    if stats:
        cv2.putText(panel, stats, (8, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return panel
