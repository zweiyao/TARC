"""把一个 episode 的任意视角合成为视频 (主视角 / 腕部 / 触觉热力图)。

    python3 demo_video.py demo/000002 --views main,wrist
    python3 demo_video.py demo/000002 --views main,wrist,tactile

时间基准取自 states.npz 的 sample_timestamps 并重采样, 不用 metadata.yaml 的
sample_hz —— 后者是设定值 30, 实测 median 只有 20.1Hz, 且有 179 帧落在 60-70ms、
3 次最长 1056ms 的停顿。按 30fps 直接编码会让画面快 1.7 倍并抹掉停顿。

触觉那两路用 marker 检测+匹配 (见 markerflow.py), 不是稠密光流 —— marker 间距
只有 10px, 稠密光流会周期性混叠。
"""

import argparse
import pathlib

import cv2
import numpy as np

from markerflow import MarkerTracker, densify

PANEL_H = 448          # 所有面板统一到这个高度再横向拼接


def resample(ts, fps):
    """按真实时间戳重采样到固定输出帧率, 返回每个输出帧对应的源帧下标。

    停顿期间会重复上一帧, 所以视频里的停顿和实际采集一样长。
    """
    t = ts - ts[0]
    out_t = np.arange(0.0, t[-1], 1.0 / fps)
    return np.searchsorted(t, out_t, side="right").clip(0, len(t) - 1), out_t


def fit(img, h=PANEL_H):
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def heat(pts, disp, shape, vmax, arrow_gain, arrow_min):
    dense = densify(pts, disp, shape)
    img = cv2.applyColorMap(
        (np.clip(np.linalg.norm(dense, axis=-1) / vmax, 0, 1) * 255).astype(np.uint8),
        cv2.COLORMAP_TURBO)
    img = fit(img)
    s = PANEL_H / shape[0]
    for (x, y), (dx, dy) in zip(pts, disp):
        # 静止噪声底约 0.09px; 低于 arrow_min 不画, 否则放大后满屏乱指
        if dx * dx + dy * dy < arrow_min ** 2:
            continue
        cv2.arrowedLine(img, (int(x * s), int(y * s)),
                        (int((x + dx * arrow_gain) * s), int((y + dy * arrow_gain) * s)),
                        (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.3)
    return img, float(np.linalg.norm(disp, axis=1).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("episode")
    p.add_argument("--views", default="main,wrist",
                   help="逗号分隔: main, wrist, tactile")
    p.add_argument("-o", "--out")
    p.add_argument("--fps", type=float, default=20.0, help="输出帧率; 实测 median 20.1Hz")
    p.add_argument("--vmax", type=float, default=3.5, help="触觉色标上限 px")
    p.add_argument("--arrow-gain", type=float, default=5.0)
    p.add_argument("--arrow-min", type=float, default=0.5)
    p.add_argument("--ref-frames", type=int, default=30)
    p.add_argument("--quality", type=int, default=90)
    a = p.parse_args()

    ep = pathlib.Path(a.episode)
    views = [v.strip() for v in a.views.split(",") if v.strip()]
    ts = np.load(ep / "states.npz")["sample_timestamps"]
    idx, out_t = resample(ts, a.fps)
    print(f"{ep.name}: {views}  源 {len(ts)} 帧 / {ts[-1]-ts[0]:.1f}s "
          f"-> 输出 {len(idx)} 帧 @ {a.fps} fps")

    seqs, trackers = {}, {}
    for v in views:
        if v == "tactile":
            for s in ("left", "right"):
                d = ep / "tactile" / s / "rectify"
                if not d.is_dir():
                    raise SystemExit(f"没有 {d}")
                seqs[f"tactile:{s}"] = sorted(d.glob("*.jpg"))
                ref = np.mean([cv2.imread(str(f)).astype(np.float32)
                               for f in seqs[f"tactile:{s}"][:a.ref_frames]], 0)
                trackers[s] = MarkerTracker(ref.astype(np.uint8))
                print(f"  tactile {s}: {len(trackers[s].ref)} 个 marker")
        else:
            d = ep / v / "color"
            if not d.is_dir():
                raise SystemExit(f"没有 {d}")
            seqs[v] = sorted(d.glob("*.jpg"))
            print(f"  {v}: {len(seqs[v])} 帧 {cv2.imread(str(seqs[v][0])).shape}")

    out = pathlib.Path(a.out) if a.out else ep.parent / f"{ep.name}_{'-'.join(views)}.mp4"
    vw, cache = None, {}

    for k, src in enumerate(idx):
        panels = []
        for v in views:
            if v == "tactile":
                for s in ("left", "right"):
                    if cache.get(("t", s, src)) is None:
                        img = cv2.imread(str(seqs[f"tactile:{s}"][src]))
                        pts, disp = trackers[s].step(img)
                        cache[("t", s, src)] = heat(pts, disp, img.shape[:2],
                                                    a.vmax, a.arrow_gain, a.arrow_min)
                    h, m = cache[("t", s, src)]
                    panels.append(label(h.copy(), f"tactile {s}  |d| {m:.2f} px"))
            else:
                panels.append(label(fit(cv2.imread(str(seqs[v][src]))), v))
        frame = np.hstack(panels)
        cv2.putText(frame, f"{k:04d}  t={out_t[k]:5.2f}s  src#{src}",
                    (6, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 a.fps, (frame.shape[1], frame.shape[0]))
            if not vw.isOpened():
                raise SystemExit("VideoWriter 打不开")
        vw.write(frame)
        if k % 150 == 0:
            print(f"  {k}/{len(idx)}", flush=True)
        cache = {key: val for key, val in cache.items() if key[2] >= src}

    vw.release()
    print(f"-> {out}  ({out.stat().st_size / 1e6:.1f} MB, {len(idx)/a.fps:.1f}s)")


if __name__ == "__main__":
    main()
