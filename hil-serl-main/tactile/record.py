"""脚本 3: 从共享内存按固定频率取帧, 存图 + 时间戳。

    conda run --no-capture-output -n tarc python record.py            # 一直录到 Ctrl-C
    conda run --no-capture-output -n tarc python record.py --seconds 60

需要 capture.py 已在运行。Ctrl-C 干净收尾。

只存原始帧: 流场 float32 是 1.8MB/帧 (比原图还大 8 倍), 且能从原始帧 + 参考帧
完全重算, 没必要存。参考帧必须存, 否则这批数据无法复现。
"""

import argparse
import datetime
import json
import pathlib
import signal
import time

import cv2

from flow import TactileFlow
from shmframe import FrameReader, read_distinct


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shm", default="tactile_frames")
    p.add_argument("--dir", default="recordings")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--quality", type=int, default=90,
                   help="JPEG 质量; q90 约 220KB/帧, 10Hz 下 7.9 GB/小时")
    p.add_argument("--ref-frames", type=int, default=30)
    p.add_argument("--proc-size", default="640x360", help="参考帧的处理分辨率, 供离线复算")
    p.add_argument("--seconds", type=float, default=0, help="录多少秒; 0 表示直到 Ctrl-C")
    p.add_argument("--stale", type=float, default=5.0,
                   help="多少秒没有新帧就停止 (capture 重启后旧 reader 会永久冻结)")
    a = p.parse_args()

    rd = FrameReader(a.shm)
    d = pathlib.Path(a.dir) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d.mkdir(parents=True, exist_ok=True)

    # 自己采基线, 不依赖 viewer —— 三个脚本互相独立是这套设计的意义所在
    print(f"[rec] 采参考帧 ({a.ref_frames} 帧), 请保持无接触...", flush=True)
    pw, ph = (int(x) for x in a.proc_size.split("x"))
    tf = TactileFlow(size=(pw, ph))
    tf.set_reference(read_distinct(rd, a.ref_frames))
    cv2.imwrite(str(d / "reference.png"), tf.ref_bgr)
    (d / "params.json").write_text(json.dumps({
        "proc_size": [pw, ph], "channel": "R", "jpeg_quality": a.quality,
        "rec_fps": a.fps, "ref_frames": a.ref_frames,
    }, indent=2), encoding="utf-8")

    running = True

    def bye(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)

    interval = 1.0 / a.fps if a.fps > 0 else 0.0
    meta = open(d / "meta.jsonl", "w", encoding="utf-8")
    n, nbytes, last, dup = 0, 0, -1, 0
    t_start = t_fresh = time.monotonic()
    next_at = time.monotonic()
    print(f"[rec] 开始 -> {d}  {a.fps or '全速'} Hz", flush=True)

    while running:
        if interval:
            next_at += interval
            delay = next_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            elif delay < -1.0:
                next_at = time.monotonic()   # 落后太多重新对齐, 不做补偿式连拍
        got = rd.latest()
        if got is None or got[1] == last:
            if got is not None:
                dup += 1    # 相机比取帧频率还慢, 同一帧重复出现 —— 跳过而不是重复存
            # capture 重启会建新的 shm 段, 旧 reader 会永久冻结, 超时就收尾退出
            if time.monotonic() - t_fresh > a.stale:
                print(f"[rec] {a.stale}s 没有新帧, capture.py 挂了或已重启, 停止录制",
                      flush=True)
                break
            continue
        img, seq, t_wall, t_mono = got
        last = seq
        t_fresh = time.monotonic()

        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        if not ok:
            continue
        name = f"{n:06d}.jpg"
        (d / name).write_bytes(buf)
        meta.write(json.dumps({"seq": seq, "t": t_wall, "t_mono": t_mono,
                               "file": name}) + "\n")
        meta.flush()   # 边写边 flush, 中途崩了已写的部分仍可用
        n += 1
        nbytes += len(buf)
        if a.seconds and time.monotonic() - t_start >= a.seconds:
            break

    meta.close()
    rd.close()
    dt = time.monotonic() - t_start
    print(f"[rec] 结束 {d}: {n} 帧 / {dt:.1f}s = {n / max(dt, 1e-9):.2f} Hz, "
          f"{nbytes / 1e6:.1f} MB" + (f", 跳过重复帧 {dup}" if dup else ""), flush=True)


if __name__ == "__main__":
    main()
