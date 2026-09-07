"""脚本 2: 从共享内存按固定频率取帧, 光流+渲染, MJPEG 推到 9000 端口。

    conda run --no-capture-output -n tarc python viewer.py

需要 capture.py 已在运行。可以随时重启, 不影响采集和录制。
浏览器打开 http://10.192.36.184:9000
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from flow import TactileFlow, render
from shmframe import FrameReader, read_distinct

PAGE = """<!doctype html><meta charset=utf-8><title>Tactile Flow</title>
<style>
 body{background:#111;color:#ddd;font:14px/1.6 -apple-system,sans-serif;margin:0;padding:16px}
 img{max-width:100%;display:block;margin:12px 0;border:1px solid #333}
 button{background:#2a2a2a;color:#ddd;border:1px solid #444;padding:8px 16px;
        border-radius:6px;cursor:pointer;font-size:14px}
 button:hover{background:#383838}
 code{background:#222;padding:2px 5px;border-radius:3px}
 .hint{color:#888}
</style>
<h2>触觉传感器 光流位移热力图</h2>
<button onclick="fetch('/reset_ref').then(()=>msg.textContent='已重采基线')">
  重采参考帧 (须在无接触状态)</button>
<span id=msg class=hint></span>
<img src="/stream.mjpg">
<p class=hint>
左上 <code>raw</code> 原始 · 右上 <code>|cur-ref|</code> 接触掩码 ·
左下 <code>|d|</code> 位移大小/剪切 · 右下 <code>div(d)</code> 按压 (红=外散 蓝=内聚)<br>
按住不动时左下应持续亮着 — 若归零说明光流退化成了相邻帧。
</p>
"""


class Pipeline:
    def __init__(self, a):
        self.rd = FrameReader(a.shm)
        self.proc = TactileFlow(size=(a.width, a.height), ema=a.ema,
                                flip_div=a.flip_div, winsize=a.winsize)
        self.a = a
        self.interval = 1.0 / a.fps if a.fps > 0 else 0.0
        self.cond = threading.Condition()
        self.jpeg = None
        self.seq = 0
        self.stop = threading.Event()
        self.want_reset = threading.Event()

    def _baseline(self):
        self.proc.set_reference(read_distinct(self.rd, self.a.ref_frames))
        print(f"[view] 基线已建立 ({self.a.ref_frames} 帧平均)", flush=True)

    def run(self):
        self._baseline()
        last, fps, t_prev, next_at = -1, 0.0, time.time(), time.monotonic()
        t_fresh = time.monotonic()
        while not self.stop.is_set():
            # 自己按固定频率取最新帧, 天然完成抽帧, 不需要额外的丢帧逻辑
            if self.interval:
                next_at += self.interval
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_at = time.monotonic()   # 落后太多就重新对齐
            got = self.rd.latest()
            if got is None or got[1] == last:
                # capture 重启会建新的 shm 段, 而我们还映射着旧段 —— 画面会永远冻结。
                # 静默冻结比崩溃更糟, 所以超时就退出, 让外部重启。
                if time.monotonic() - t_fresh > self.a.stale:
                    print(f"[view] {self.a.stale}s 没有新帧, capture.py 挂了或已重启; "
                          f"本进程退出, 请重启", flush=True)
                    self.stop.set()
                    break
                if not self.interval:
                    time.sleep(0.002)
                continue
            img_in, last = got[0], got[1]
            t_fresh = time.monotonic()

            if self.want_reset.is_set():
                self.want_reset.clear()
                self._baseline()
                continue

            fields = self.proc.process(img_in)
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            img = render(fields, vmax=self.a.vmax, div_vmax=self.a.div_vmax,
                         stats=f"{fps:4.1f} fps")
            if self.a.out_scale != 1.0:
                img = cv2.resize(img, None, fx=self.a.out_scale, fy=self.a.out_scale,
                                 interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.a.quality])
            if not ok:
                continue
            with self.cond:
                self.jpeg = buf.tobytes()
                self.seq += 1
                self.cond.notify_all()

    def wait_frame(self, last_seq, timeout=5.0):
        with self.cond:
            if not self.cond.wait_for(lambda: self.seq != last_seq, timeout):
                return None, last_seq
            return self.jpeg, self.seq


def make_handler(pipe):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass  # 每帧一条日志会刷屏

        def _send(self, body, ctype="text/html; charset=utf-8"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(PAGE.encode())
            elif path == "/reset_ref":
                pipe.want_reset.set()
                self._send(json.dumps({"ok": True}).encode(), "application/json")
            elif path == "/stream.mjpg":
                self.stream()
            else:
                self.send_error(404)

        def stream(self):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while not pipe.stop.is_set():
                    jpeg, seq = pipe.wait_frame(seq)
                    if jpeg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpeg)).encode()
                                     + b"\r\n\r\n" + jpeg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass  # 客户端关页面

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shm", default="tactile_frames")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--fps", type=float, default=10.0, help="取帧频率; 0 表示全速")
    p.add_argument("--width", type=int, default=640, help="光流处理分辨率")
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--out-scale", type=float, default=0.75, help="面板输出缩放")
    p.add_argument("--vmax", type=float, default=3.0, help="位移色标上限 (px)")
    p.add_argument("--div-vmax", type=float, default=0.12,
                   help="散度色标上限; 无接触噪声 p99 约 0.031")
    p.add_argument("--ema", type=float, default=0.5)
    p.add_argument("--winsize", type=int, default=21)
    p.add_argument("--ref-frames", type=int, default=30)
    p.add_argument("--quality", type=int, default=80, help="串流 JPEG 质量")
    p.add_argument("--stale", type=float, default=5.0,
                   help="多少秒没有新帧就退出 (capture 重启后旧 reader 会永久冻结)")
    p.add_argument("--flip-div", action="store_true")
    a = p.parse_args()

    pipe = Pipeline(a)
    threading.Thread(target=pipe.run, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(pipe))
    srv.daemon_threads = True
    print(f"[view] http://0.0.0.0:{a.port}  取帧 {a.fps or '全速'} Hz", flush=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        while not pipe.stop.is_set():   # 取帧线程判定 stale 后会置位, 主线程随之退出
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop.set()
        with pipe.cond:
            pipe.cond.notify_all()
        srv.shutdown()
        pipe.rd.close()


if __name__ == "__main__":
    main()
