"""脚本 1: 独占相机, 全速采集写入共享内存。不做任何处理。

    conda run --no-capture-output -n tarc python capture.py

这是唯一打开 /dev/video0 的进程。viewer.py 和 record.py 都从共享内存读, 可以
随意起停而不影响采集。
"""

import argparse
import signal
import time

import cv2

from shmframe import FrameWriter


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--shm", default="tactile_frames")
    p.add_argument("--slots", type=int, default=8)
    a = p.parse_args()

    cap = cv2.VideoCapture(a.device, cv2.CAP_V4L2)
    # 必须用 MJPG: 默认 YUYV 在 1080p 下被 USB 带宽限制到 5fps
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"打不开 /dev/video{a.device}")

    ok, f = cap.read()
    if not ok:
        raise SystemExit("相机打开了但读不出帧")
    h, w = f.shape[:2]
    w_shm = FrameWriter(a.shm, h, w, a.slots)
    print(f"[cap] {w}x{h} -> shm '{a.shm}' "
          f"({a.slots} 槽 / {w_shm.nbytes / 1e6:.0f} MB)", flush=True)

    running = True

    def bye(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)

    fps, t_prev, t_log = 0.0, time.time(), 0.0
    while running:
        ok, img = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        # 时间戳在这里取, 紧挨着 read() 返回 —— 下游取到的会含排队延迟
        now = time.time()
        w_shm.publish(img, now, time.monotonic())
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
        t_prev = now
        if now - t_log > 5.0:
            print(f"[cap] seq={w_shm.seq} {fps:.1f} fps", flush=True)
            t_log = now

    cap.release()
    w_shm.close()
    print("[cap] 已退出, 共享内存已释放", flush=True)


if __name__ == "__main__":
    main()
