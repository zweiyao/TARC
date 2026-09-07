> **TARC 里的状态**
>
> 这份 README 以下部分是从 `/Users/zweiyao/work/tactile` 原样拷来的，描述的是那个
> 独立项目。拷进 TARC 时只带了代码（8 个文件，68KB），没带数据——源目录 2.3G，
> 其中 `demo.zip` 1.2GB、`demo/` 是录制帧、`picks/` 是挑出来的热力图样例，
> 都留在原处。
>
> **新增的只有 `synth.py`，它是占位。** 传感器还没接上，而录制的散斑帧在
> `10.192.36.184` 上（当前不可达），所以它用程序生成散斑：造一张随机 RGB 散斑做
> 基线，施加一个已知的高斯位移鼓包当作按压。**从这两帧往后全是真的**——它们走的
> 是 `flow.py` 的 Farneback，和真实相机帧完全一样的路径。
>
> 生成而非回放有一个录制做不到的好处：位移是自己施加的，量已知，所以能反向断言
> 光流真的算对了。实测施加 2.500px、还原 2.478px（误差 0.9%），静止帧噪声底
> mean 0.0022 / p99 0.0067。
>
> 接上真实传感器后要换掉的是 `synth.py` 里的 `press()`：参考帧改成
> `set_reference(read_distinct(rd, 30))` 从共享内存取 30 帧无接触画面，当前帧改成
> `rd.latest()`。`magnitude()` 和 `heatmap()` 不用动。
>
> 热力图输出 `(240,320,3)` uint8 BGR，喂给观测里的 `tactile` key。原生是
> 640×360，缩到 240×320 是为了控制 `SpatialLearnedEmbeddings` 的参数量
> （360×640 会到 98.3 万，是相机的 15 倍）。
>
> 消费方：`examples/test/test_temporal_obs_dataflow.py`。
>
> 本次没有拷 `sync.sh`——它的同步目标是 `10.192.36.184`，与 TARC 的远程
> （`10.176.53.118`）不是同一台。

---

# 触觉传感器光流热力图

把相机式触觉传感器的凝胶形变实时渲染成热力图，通过网页流查看。

## 架构

三个独立进程，通过共享内存传帧。相机只能被一个进程打开，所以由 `capture.py` 独占：

```
capture.py   独占 /dev/video0, 全速采集 -> 共享内存         ~30 fps
     │  /dev/shm/tactile_frames  (8 槽环形缓冲, 50 MB)
     │  每帧带 seq + t_wall + t_mono
     ├──> viewer.py   按 10Hz 取帧 -> 光流+渲染 -> MJPEG :9000
     └──> record.py   按 10Hz 取帧 -> JPEG + meta.jsonl
```

**好处是三者可以独立起停。** viewer 崩了或要改参数重启，不影响采集和正在进行的录制——实测录制途中杀掉并重启 viewer，`capture` 全程 29.9 fps 无中断，`record` 仍是精确 10.00 Hz。

共享内存用环形缓冲加序号校验，无锁：writer 写完槽位才发布全局序号，reader 拷完再校验序号没变。writer 两帧间隔 33ms、reader 拷一帧 6.2MB 约 1ms、8 个槽位，时序余量极大。

## 原理

凝胶表面是**随机 RGB 散斑**（不是 marker 点阵，也不是 GelSight 的三向彩色 LED）。散斑等价于实验力学里数字图像相关（DIC）刻意喷到试件表面的图案，让每个局部窗口都有唯一可匹配的特征，因此**稠密光流可解且精度很高**——已知平移 0.5/1/2/4 px，实测误差均 < 0.01 px。

```
相机 1920×1080 (MJPG)
  └─ 降采样 → 640×360
      └─ 取 R 通道 → CLAHE
          ├─ ref: 启动时 30 帧无接触平均
          └─ Farneback(ref → cur)      ← 相对参考帧, 不是相邻帧
              └─ (u, v) 稠密位移场
                  ├─ mag = √(u²+v²)      位移大小 / 剪切
                  ├─ div = ∂u/∂x+∂v/∂y   按压 (无量纲应变)
                  └─ |cur − ref|          接触掩码
```

> **核心约束**：光流必须相对一个**固定的无接触参考帧**计算 `flow(ref → cur)`，而不是相邻帧 `flow(prev → cur)`。触觉信号表达的是形变**状态**而非速度——按住不动时相邻帧光流会归零，热力图会灭掉。

`mag` 对切向抹动和法向按压都响应；`div` 只对法向按压响应（压下去凝胶被撑开，散斑径向外散）。两者结合才能区分「横着抹」和「竖着按」。

## 环境

**不需要安装任何东西。** 远端 `10.192.36.184` 的 conda 环境 `tarc` 已包含全部依赖：

| | |
|---|---|
| Python | 3.10.20 |
| opencv-python | 4.10.0 |
| numpy | 1.26.4 |

服务端只用 Python 标准库做 HTTP，无 Flask/FastAPI 等额外依赖。

硬件：触觉传感器接在 `/dev/video0`（UVC `05a3:2b01`），支持 MJPG 1920×1080@30fps。

## 部署

本地编辑，推到远端执行：

```bash
./sync.sh
```

- 本地：`/Users/zweiyao/work/tactile`
- 远端：`zwy@10.192.36.184:~/work/embodied_ai/tactile`

## 启动

三个脚本都在 `~/work/embodied_ai/tactile/` 下、`tarc` 环境里跑。

**1. 采集（必须先起，且一直开着）**

```bash
ssh zwy@10.192.36.184 'source ~/miniconda3/etc/profile.d/conda.sh; cd ~/work/embodied_ai/tactile; nohup conda run --no-capture-output -n tarc python capture.py > /tmp/cap.log 2>&1 &'
```

**2. 查看（可随时起停）**

```bash
ssh zwy@10.192.36.184 'source ~/miniconda3/etc/profile.d/conda.sh; cd ~/work/embodied_ai/tactile; nohup conda run --no-capture-output -n tarc python viewer.py > /tmp/view.log 2>&1 &'
```

浏览器打开 **http://10.192.36.184:9000**，启动约 10 秒（等采基线）。页面是 2×2 面板：

| | |
|---|---|
| 左上 `raw` 原始图 | 右上 `\|cur−ref\|` 接触掩码 |
| 左下 `\|d\|` 位移大小 + 箭头 | 右下 `div(d)` 按压（红=外散 蓝=内聚）|

**「重采参考帧」**按钮在无接触状态下重建基线。凝胶有蠕变、相机有曝光漂移，跑久了背景会浮起来，这时重采一下。

**3. 录制（可随时起停）**

```bash
ssh zwy@10.192.36.184 'source ~/miniconda3/etc/profile.d/conda.sh; cd ~/work/embodied_ai/tactile; conda run --no-capture-output -n tarc python record.py --seconds 60'
```

不带 `--seconds` 就一直录到 Ctrl-C。开始前会先采 30 帧参考帧，**这几秒要保持无接触**。

### 启停顺序很重要

`capture.py` 重启会创建**新的**共享内存段，而已在运行的 viewer/record 仍映射着旧段，会永远看到冻结画面。所以 **重启 capture 后必须也重启 viewer 和 record**。

两个 reader 都有失效检测：5 秒没有新帧就打印原因并退出（`--stale` 可调），不会静默冻结。

### 停止

不要用 `pkill -f capture.py` —— ssh 命令行里含有脚本名，pkill 会把 ssh 会话自己也杀掉。按占用的资源找：

```bash
ssh zwy@10.192.36.184 'for p in $(fuser /dev/video0 2>/dev/null); do kill $p; done; P=$(ss -tlnp | grep ":9000" | grep -oP "pid=\K[0-9]+" | head -1); [ -n "$P" ] && kill $P'
```

`record.py` 直接 Ctrl-C，会干净收尾。

## 录制

`record.py` 是独立进程，起停不影响采集和显示。实测 **10.00 Hz、间隔 mean 100.0ms std 4.4ms、无重复帧**。

### 落盘结构

```
recordings/2026-09-02_17-46-38/
    000000.jpg        原始帧 1920×1080, JPEG q90
    000001.jpg
    ...
    meta.jsonl        每帧一行, 边写边 flush
    reference.png     当时用的参考帧 (640×360)
    params.json       处理参数
```

`meta.jsonl` 每行：

```json
{"seq": 581, "t": 1788342398.8151636, "t_mono": 96516.957853993, "file": "000000.jpg"}
```

- `seq` — 相机帧号，10Hz 录制时相邻两条差约 3（源是 26fps）
- `t` — `time.time()`，绝对时间，用于和机器人状态、力传感器对齐
- `t_mono` — `time.monotonic()`，不受 NTP 跳变影响，算帧间隔用这个

时间戳是在采集线程里 `cap.read()` 返回后**立即**取的，不是写盘时间，所以不含排队延迟。

**`reference.png` 必须保留** —— 没有它这批数据无法复现处理结果。

### 为什么只存原始帧

流场 `float32 640×360×2` 是 **1.8 MB/帧**，比原图还大 8 倍，而且可以从原始帧 + 参考帧完全重算。所以不存，离线想换参数重算随时可以。

### 为什么是 JPEG 不是 PNG

实测 1920×1080 编码：

| 格式 | 编码耗时 | 大小/帧 | 10Hz 时 |
|---|---|---|---|
| JPEG q95 | 3.0 ms | 349 KB | 12.0 GB/小时 |
| **JPEG q90** | **2.7 ms** | **220 KB** | **7.9 GB/小时** |
| JPEG q80 | 2.5 ms | 130 KB | 4.7 GB/小时 |
| PNG lv1 | 80.2 ms | 2.3 MB | 82.8 GB/小时 |

PNG 编码 80ms 会直接压垮帧率；更重要的是**相机输出的本来就是 MJPEG**，数据已经被 JPEG 压过一次，用 PNG「无损」保存等于无损地保留 JPEG 伪影——零信息增益。

实测录制体积约 143 KB/帧（比基准图略小），10Hz 下约 **5.1 GB/小时**。磁盘 1.7T 可用，够录约 330 小时。

### 取帧频率

`--fps` 默认 10（`viewer.py` 和 `record.py` 各自独立设置）。每个消费者跑自己的时钟，到点去拿共享内存里的最新帧——**抽帧是自然发生的**，不需要额外的丢帧逻辑。

实测 **10.00 Hz，间隔 mean 100.0ms / std 4.4ms**（min 68 / max 105）。抖动来自源只有 30fps（33ms 一帧），只能在这些时刻里挑，但平均值精确。

`--fps 0` 表示不限速，按相机全速跑。

### 重复帧与失效

消费频率高于相机帧率时会拿到同一帧，`record.py` 按 `seq` 去重、跳过而不是重复存，收尾时报告跳过数。

5 秒没有新帧就判定失效并退出（`--stale`）。这覆盖两种情况：`capture.py` 挂了，或者它重启后建了新的共享内存段而本进程还映射着旧段——后者会导致**画面永久冻结**，静默冻结比崩溃更糟。

## 获取触觉图像

只要 `capture.py` 在跑，**任何脚本都能从共享内存直接读帧，不用停任何服务**——这是三进程拆分带来的主要便利。

### A. 从共享内存读（推荐）

脚本要和 `shmframe.py` 放在同一目录：Python 把**脚本所在目录**加进 `sys.path`，不是当前工作目录，放 `/tmp` 里跑会 `ModuleNotFoundError`。

```python
from flow import TactileFlow
from shmframe import FrameReader, read_distinct

rd = FrameReader("tactile_frames")
tf = TactileFlow(size=(640, 360))
tf.set_reference(read_distinct(rd, 30))      # 必须在无接触状态

img, seq, t_wall, t_mono = rd.latest()       # img 是 1920x1080 BGR 原始帧
r = tf.process(img)
rd.close()
```

`r` 是一个 dict：

| 键 | 形状 | 含义 |
|---|---|---|
| `r["mag"]` | (360, 640) float32 | 位移大小，单位 px |
| `r["div"]` | (360, 640) float32 | 散度，无量纲应变 |
| `r["flow"]` | (360, 640, 2) float32 | 位移场 (u, v)，单位 px |
| `r["small"]` | (360, 640, 3) uint8 | 原始降采样图 BGR |
| `r["diff"]` | (360, 640) uint8 | 与参考帧的通道最大差 |

渲染成面板图用 `flow.render(r)`。只要原始帧不做处理，`rd.latest()` 就够了，不必碰 `TactileFlow`。

### B. 从 9000 端口的流里存图

只是想截一张图看看：

```bash
curl -s -m 3 http://10.192.36.184:9000/stream.mjpg | \
  python3 -c "import sys;d=sys.stdin.buffer.read();c=d.split(b'--frame')[-2];i=c.find(b'\xff\xd8');open('panel.jpg','wb').write(c[i:c.rfind(b'\xff\xd9')+2])"
```

或在代码里读流：

```python
import cv2
cap = cv2.VideoCapture("http://10.192.36.184:9000/stream.mjpg")
ok, panel = cap.read()          # (540, 960, 3)
raw = panel[:270, :480]         # 左上角 = 原始图
```

拿到的是**渲染后的面板**，经过降采样、伪彩映射和 JPEG 压缩。**不要拿它做定量分析**，那种情况用方式 A。

## 参数

**`capture.py`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--width/--height` | 1920/1080 | 相机分辨率 |
| `--shm` | tactile_frames | 共享内存段名 |
| `--slots` | 8 | 环形缓冲槽位数（每槽 6.2 MB） |

**`viewer.py`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fps` | 10 | 取帧频率；0 = 全速 |
| `--port` | 9000 | |
| `--vmax` | 3.0 | 位移色标上限（px）。按下去整片饱和就调大 |
| `--div-vmax` | 0.12 | 散度色标上限。无接触噪声 p99 约 0.031 |
| `--width/--height` | 640/360 | 光流处理分辨率。见下方性能表 |
| `--out-scale` | 0.75 | 面板输出缩放，只影响串流带宽 |
| `--ema` | 0.5 | 位移场时间平滑系数，调小更稳但更迟钝 |
| `--winsize` | 21 | Farneback 窗口。散斑颗粒明显变大/变小时要跟着调 |
| `--flip-div` | off | 翻转散度正负号 |
| `--stale` | 5.0 | 多少秒没有新帧就退出 |

**`record.py`**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--fps` | 10 | 录制频率 |
| `--dir` | recordings | 输出根目录 |
| `--quality` | 90 | JPEG 质量 |
| `--seconds` | 0 | 录多少秒；0 = 直到 Ctrl-C |
| `--ref-frames` | 30 | 基线平均帧数 |
| `--stale` | 5.0 | 多少秒没有新帧就停止 |

色标上下限是**固定**的而非自动归一化，这样不同时刻的图才可比。

## 性能

| 处理分辨率 | 光流耗时 | 可达帧率 |
|---|---|---|
| 960×540 | 74 ms | ~12 fps |
| **640×360** | **31 ms** | **~30 fps** ← 默认 |
| 480×270 | 18 ms | ~30 fps |

降到 640×360 不损失精度（已知平移测试误差仍 < 0.01 px）。默认 10Hz 消费时光流只占 31% 的时间预算，余量很大。

实测：`capture` 29.9 fps、`viewer` 10.0 fps、`record` 10.00 Hz，三者同时运行互不影响。

无接触噪声底：`|d|` mean 0.065 / p99 0.25 px；`div` mean 0.0063 / p99 0.031。

## 排错

**帧率只有 5 fps** —— 相机没设成 MJPG。默认的 YUYV 格式在 1080p 下被 USB 带宽卡到 201 ms/帧；MJPG 是 33 ms/帧。这是本项目最大的性能陷阱。

**按住不动时热力图灭掉** —— 光流退化成了相邻帧。必须是 `flow(ref → cur)`。

**背景整体浮起来 / 松开后有残影** —— 凝胶蠕变或曝光漂移，点页面上的「重采参考帧」。

**散度面板全是饱和红蓝** —— `--div-vmax` 太小，或求导前没平滑位移场（求导会放大噪声，顺序不能反）。

**光流噪声大** —— 先检查**镜头对焦**，这比调算法参数有效得多。凝胶纹理的绝对对比度只有约 3/255，失焦会直接吃掉信号。

**浏览器连不上 9000** —— 退回 SSH 隧道：

```bash
ssh -L 9000:localhost:9000 zwy@10.192.36.184
```

然后访问 `http://localhost:9000`。

**`ModuleNotFoundError: No module named 'flow'`** —— 脚本没和 `flow.py` 放在同一目录。

**画面冻结不动 / `共享内存里没有帧`** —— `capture.py` 没在跑，或者它重启过而 viewer/record 还映射着旧段。重启 capture 后必须重启另外两个。

**`打不开 /dev/video0`** —— 相机被别的进程占着。`fuser -v /dev/video0` 看是谁；常见是旧的 `capture.py` 没退干净。

## 已知限制

- `capture.py` 重启后 viewer/record 不会自动重连（会检测到失效并退出，需要外部重启）。
- 共享内存只保留最新的 8 帧，消费者落后太多会丢帧而不是排队——这是刻意的，避免延迟累积。
- 端口 9000 在局域网内**无认证**，任何能访问这台机器的人都能看到视频流。
- 白光照明，做不了光度立体深度重建（那需要多向彩色 LED）。
- 位移单位是像素，不是物理长度或力，需要标定才能换算。
