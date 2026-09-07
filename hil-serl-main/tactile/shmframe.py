"""进程间共享相机帧的环形缓冲。

一个 writer, 多个 reader, 无锁:
  writer 先把槽位 seq 置 -1 (标记正在写), 写完数据再写回真实 seq, 最后才发布全局 seq;
  reader 读全局 seq -> 定位槽位 -> 校验槽位 seq -> 拷贝 -> 再校验一次没被覆盖。

这不是形式化严格的无锁协议 (依赖 x86 的存储顺序和 numpy 赋值的原子性), 但时序余量
极大: writer 两帧间隔 38ms, reader 拷一帧 6.2MB 约 1ms, 加上 8 个槽位, 实际不可能撞上。
校验失败时重试即可。
"""

import time

import numpy as np
from multiprocessing import resource_tracker, shared_memory

MAGIC = 0x54414331  # "TAC1"
_CTRL = 8           # 控制区 int64 个数: [magic, h, w, slots, write_seq, -, -, -]


def _layout(h, w, slots):
    off_seq = _CTRL * 8
    off_ts = off_seq + slots * 8
    off_data = off_ts + slots * 16
    return off_seq, off_ts, off_data, off_data + slots * h * w * 3


class FrameWriter:
    def __init__(self, name, h, w, slots=8):
        self.slots = slots
        off_seq, off_ts, off_data, total = _layout(h, w, slots)
        try:  # 清理上次崩溃残留的段, 否则 create=True 会 FileExistsError
            old = shared_memory.SharedMemory(name=name)
            old.close()
            old.unlink()
        except FileNotFoundError:
            pass
        self.shm = shared_memory.SharedMemory(name=name, create=True, size=total)
        b = self.shm.buf
        self.ctrl = np.ndarray(_CTRL, np.int64, buffer=b)
        self.seqs = np.ndarray(slots, np.int64, buffer=b, offset=off_seq)
        self.ts = np.ndarray((slots, 2), np.float64, buffer=b, offset=off_ts)
        self.data = np.ndarray((slots, h, w, 3), np.uint8, buffer=b, offset=off_data)
        self.seqs[:] = -1
        self.ctrl[:] = 0
        self.ctrl[:4] = (MAGIC, h, w, slots)
        self.seq = 0
        self.nbytes = total

    def publish(self, img, t_wall, t_mono):
        self.seq += 1
        i = self.seq % self.slots
        self.seqs[i] = -1            # 槽位正在写
        self.data[i] = img
        self.ts[i] = (t_wall, t_mono)
        self.seqs[i] = self.seq      # 槽位就绪
        self.ctrl[4] = self.seq      # 最后发布: reader 看到它时数据一定完整
        return self.seq

    def close(self):
        self.ctrl = self.seqs = self.ts = self.data = None
        self.shm.close()
        self.shm.unlink()


class FrameReader:
    def __init__(self, name):
        self.shm = shared_memory.SharedMemory(name=name)
        # reader 不拥有这个段, 但 Python 会把它登记进 resource_tracker,
        # 退出时报 "leaked shared_memory" 并擅自 unlink, 把 writer 的段删掉。
        resource_tracker.unregister(self.shm._name, "shared_memory")
        b = self.shm.buf
        self.ctrl = np.ndarray(_CTRL, np.int64, buffer=b)
        if int(self.ctrl[0]) != MAGIC:
            raise RuntimeError(f"共享内存 {name} 不是 tactile 帧缓冲")
        h, w, self.slots = int(self.ctrl[1]), int(self.ctrl[2]), int(self.ctrl[3])
        off_seq, off_ts, off_data, _ = _layout(h, w, self.slots)
        self.seqs = np.ndarray(self.slots, np.int64, buffer=b, offset=off_seq)
        self.ts = np.ndarray((self.slots, 2), np.float64, buffer=b, offset=off_ts)
        self.data = np.ndarray((self.slots, h, w, 3), np.uint8, buffer=b, offset=off_data)
        self.shape = (h, w)

    def latest(self, retries=5):
        """返回 (img, seq, t_wall, t_mono); 还没有帧或反复撞上写入时返回 None。"""
        for _ in range(retries):
            s = int(self.ctrl[4])
            if s <= 0:
                return None
            i = s % self.slots
            if int(self.seqs[i]) != s:
                continue
            img = self.data[i].copy()
            t_wall, t_mono = float(self.ts[i, 0]), float(self.ts[i, 1])
            if int(self.seqs[i]) == s:   # 拷贝期间没被覆盖
                return img, s, t_wall, t_mono
        return None

    def close(self):
        self.ctrl = self.seqs = self.ts = self.data = None
        self.shm.close()


def read_distinct(rd, n, timeout=15.0):
    """按帧到达速度取 n 张不同的帧。

    建基线时不该按消费频率慢慢等 —— 10Hz 取 30 帧要 3 秒, 全速只要 1.2 秒。
    """
    out, last, t0 = [], -1, time.time()
    while len(out) < n and time.time() - t0 < timeout:
        got = rd.latest()
        if got and got[1] != last:
            last = got[1]
            out.append(got[0])
        else:
            time.sleep(0.002)
    if not out:
        raise SystemExit("共享内存里没有帧 —— capture.py 在跑吗?")
    return out
