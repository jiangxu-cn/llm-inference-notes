# 第 2 讲 · 第一个可信的 benchmark

> W1 / L02 · 预计 30 min 理解 + 60–90 min 实操
> 通过标准：`bench.py` 跑通 + 说清"为什么不能不同步就计时" + 自测 3 题

---

## ① 依赖门（1 min）

**✅ 现在只需要补这些：**
1. 会用 `time.perf_counter()`（Python 标准库）
2. 知道 `torch.cuda.synchronize()` 是"等 GPU 干完"
3. 知道什么是**中位数**

**⏸️ 现在不学也没影响：**
`cudaEvent` 的底层实现、CUDA Stream 的并发模型、PCIe 传输、CUPTI —— 全部后面才用。
统计学（分布、置信区间）也不用，只要会排序取中间那个。

**🔜 什么时候会回来：**
`nsys` / `ncu` 在第 1 讲已经装好，第 3 讲开始用。本讲的 `bench_event()` 函数会在**后面 13 周里被反复调用**，是你所有性能数字的来源。

---

## ② 直觉先行（6 min）

把 GPU 想成**一家只接外卖单的餐厅**：

```
你（CPU）                餐厅（GPU）
  │
  ├─ 下单 ───────────────▶ 订单进了队列
  │  ◀── 立刻返回"好的"      后厨才开始做
  │
  ├─ 你又下单 ────────────▶ 排队
  ├─ 你又下单 ────────────▶ 排队
  │
  └─ "出餐口集合！" ───────▶ 你站到出餐口等，全部做完才走
      （synchronize）
```

**关键点：`下单`这个动作本身只要几微秒，`做菜`要几毫秒。**

所以：

```python
t0 = time.perf_counter()
x = a @ b               # ← 只是"下单"，GPU 还没开始做
t1 = time.perf_counter()
print(t1 - t0)          # ← 测到的是"下单耗时"，不是"做菜耗时"
```

**这不是快慢问题，是测错了对象。** 本机实测：这样测出来 `0.023 ms`，真实值是 `6.09 ms` —— **低估了 265 倍**。

---

## ③ 原理（8 min）

### 3.1 三个必须记住的事实

| 事实 | 含义 |
|---|---|
| **kernel launch 是异步的** | CPU 提交完就返回，不等 GPU |
| **第一次调用特别慢** | CUDA context 初始化、cuBLAS handle 创建、kernel 首次加载（lazy module loading） |
| **单次测量噪声很大** | 别的进程、显存时钟、GPU 温度、驱动后台任务都会影响 |

### 3.2 warmup：本机实测（这一步不能省）

```
iter 0:   204.275 ms     ← 第一次，慢 33 倍
iter 1:     6.158 ms
iter 2:     6.185 ms
iter 3:     6.162 ms
iter 4:     6.182 ms     ← 从这里开始才是"真实性能"
iter 5:     6.547 ms
```

**第一次调用 204 ms，稳态只要 6.16 ms。** 如果你不做 warmup 直接测，第一个样本会污染整个平均值 —— 30 次里混进一个 204 ms，平均值直接偏大 6 ms（翻倍）。

> **规则：warmup 至少 10 次，然后才开始计时。**

### 3.3 三个统计量，分别回答什么问题

| 统计量 | 回答的问题 | 什么时候用 |
|---|---|---|
| **min** | 「无干扰时最快能到多少」 | 探上限、和理论峰值比 |
| **median** | 「典型情况是多少」 | **默认用这个**，抗离群值 |
| **mean** | 「总共花了多少 / 次数」 | 算总账、算成本 |
| max | 「被干扰得多严重」 | 诊断：max 远大于 median = 有干扰 |

**为什么默认 median 而不是 mean？** mean 会把个别慢样本的偏差平均进结果。本机实测（30 次）：

```
min    :   5.7358 ms
median :   5.8833 ms   ← 默认
mean   :   5.8791 ms
max    :   6.0136 ms
```

这次 mean 和 median 很接近，是因为**没有干扰**。一旦有别的进程抢 GPU，mean 会被拉高而 median 基本不动。

### 3.4 换算成"有物理意义的数字"

**只报 ms 是不够的** —— ms 不能告诉你优化空间还有多大。必须换算成下面两种：

```
算力（compute-bound）:  TFLOPS = 2 × M × N × K / 时间
带宽（memory-bound）:   GB/s   = 读写总字节数 / 时间
```

本机 4096³ FP32 matmul 实测：

| | 实测 | 理论峰值 | 达成率 |
|---|---|---|---|
| matmul | **21.0 TFLOPS** | ~30 TFLOPS（FP32） | 70% |
| 256 MB copy | **577.8 GB/s** | ~672 GB/s | 86% |

> **有了这两个数，你才知道"还能快多少"。** 21/30 = 70% 说明还有优化空间；如果实测已经 95%，那就别在这上面浪费时间了。
>
> 👉 这正是 **W2 要讲的 Roofline 模型**的雏形。现在先建立习惯：**每个性能数字都要带单位**。

---

## ④ 实操（12 min）

代码在 [`code/bench.py`](code/bench.py)，92 行。在 WSL 里跑：

```bash
~/work/venv/bin/python ~/work/bench.py
```

**完整代码：**

```python
#!/usr/bin/env python3
"""L02 · 第一个可信的 benchmark"""
import time
import torch

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False   # 保证测的是真 FP32，不是 TF32

print(f"GPU: {torch.cuda.get_device_name()}  capability={torch.cuda.get_device_capability()}")

# ============ 1. 为什么开头几次调用特别慢 ============
print("【1】开头几次调用为什么特别慢（warmup 的意义）")
a = torch.randn(4096, 4096, device=DEV)
b = torch.randn(4096, 4096, device=DEV)

def matmul():
    return a @ b

for i in range(6):
    t0 = time.perf_counter()
    matmul()
    torch.cuda.synchronize()
    print(f"    iter {i}: {(time.perf_counter() - t0) * 1e3:9.3f} ms")

# ============ 2. 三种计时方法对比 ============
def bench_naive(fn, rep=50):
    """❌ 错误示范：不 synchronize"""
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep * 1e3

def bench_sync(fn, warmup=10, rep=50):
    """✅ perf_counter + synchronize，取中位数"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]

def bench_event(fn, warmup=10, rep=50):
    """CUDA Event 写法（本机实测有偏差，见附录）"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends   = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]

n_naive = bench_naive(matmul)
n_sync  = bench_sync(matmul)
n_event = bench_event(matmul)

print("【2】同一个 matmul，三种计时方法")
print(f"    ❌ 不 synchronize    : {n_naive:9.4f} ms   <- 测到的是提交时间")
print(f"    ✅ perf_counter+sync : {n_sync:9.4f} ms")
print(f"    ⚠️ CUDA Event        : {n_event:9.4f} ms")
print(f"    -> naive 把真实时间低估了 {n_sync / n_naive:.0f} 倍")

# ============ 3. 换算成有物理意义的数字 ============
flops = 2 * 4096 ** 3
print(f"    matmul 4096^3 : {flops / (n_sync / 1e3) / 1e12:6.1f} TFLOPS FP32 (理论峰值 ~30)")

n = 64 * 1024 * 1024                      # 64M 个 float = 256 MB
x = torch.empty(n, device=DEV)
y = torch.empty(n, device=DEV)
ms_copy = bench_sync(lambda: y.copy_(x))
gb = 2 * n * 4 / 1e9                      # 读 256MB + 写 256MB
print(f"    copy 256MB    : {gb / (ms_copy / 1e3):6.1f} GB/s      (理论 ~672)")

# ============ 4. 中位数 vs 平均值 vs 最小值 ============
ts = []
for _ in range(30):
    t0 = time.perf_counter()
    matmul()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
ts.sort()
mean = sum(ts) / len(ts)
print("【4】取哪个统计量？（30 次重复）")
print(f"    min    : {ts[0]:8.4f} ms   <- 最接近「无干扰」的真实性能")
print(f"    median : {ts[len(ts) // 2]:8.4f} ms   <- 默认用这个，抗离群值")
print(f"    mean   : {mean:8.4f} ms")
print(f"    max    : {ts[-1]:8.4f} ms")
```

**预期输出（本机实跑）：**

```
GPU: NVIDIA GeForce RTX 5070  capability=(12, 0)

【1】开头几次调用为什么特别慢（warmup 的意义）
    iter 0:   204.275 ms
    iter 1:     6.158 ms
    iter 2:     6.185 ms
    iter 3:     6.162 ms
    iter 4:     6.182 ms
    iter 5:     6.547 ms

【2】同一个 matmul，三种计时方法
    ❌ 不 synchronize    :    0.0230 ms   <- 测到的是提交时间
    ✅ perf_counter+sync :    6.0878 ms
    ✅✅ CUDA Event       :    6.5573 ms
    -> naive 把真实时间低估了 265 倍

【3】换算成物理量（这才是能写进报告的数字）
    matmul 4096^3 :   21.0 TFLOPS FP32 (理论峰值 ~30)
    copy 256MB    :  577.8 GB/s      (理论 ~672)

【4】取哪个统计量？（30 次重复）
    min    :   5.7358 ms
    median :   5.8833 ms
    mean   :   5.8791 ms
    max    :   6.0136 ms
```

---

## ⑤ 语法与坑点（2 min）

| 语法 / 坑 | 为什么 | 最容易错在哪 |
|---|---|---|
| `torch.cuda.synchronize()` | CUDA 是异步的，必须显式等待 | 只在**循环结束后**调一次 ≠ 每次调用后调。放在循环外就变成 `bench_naive` |
| `time.perf_counter()` | 单调时钟，不受系统时间调整影响 | 别用 `time.time()`（可能被 NTP 回调） |
| `torch.cuda.Event(enable_timing=True)` | **不写 `enable_timing=True` 就永远是 0** | 默认是 `False`，`elapsed_time()` 返回 0 |
| `torch.backends.cuda.matmul.allow_tf32` | TF32 比 FP32 快约 2 倍但精度低。**不显式关掉，你测到的可能是 TF32** | 这是"数字对不齐"的最常见原因之一 |
| `sorted(ts)[len(ts)//2]` | 取中位数 | `statistics.median()` 更清晰，但**别用 `numpy.median`** —— 它要装 numpy 且对 list 有额外开销 |
| `2 * n * 4 / 1e9` | 换算字节数：`n` 个元素 × 4 字节 × 2（读+写） | **忘记 ×2 是最常见的带宽算错原因**，会让你的带宽凭空少一半 |

---

## ⑥ 自测 3 题（1 min）

1. `bench_naive` 测出 0.023 ms，`bench_sync` 测出 6.09 ms。**为什么前者不是"更快"，而是"错的"？** 提示：想想 265 倍这个比例意味着什么。
2. 你做了一次 30 次重复的测量，`min = 5.74 ms`、`max = 18.3 ms`。**这说明什么？你应该报告哪个数，为什么？**
3. 一个 kernel 读写 512 MB 数据，耗时 1.8 ms。**它的带宽是多少 GB/s？**（先自己想，再核对：`512 / 1.8e-3` 还是 `2 × 512 / 1.8e-3`？这两者的区别是什么？）

---

## 🔬 附录：本机实测的一个重要发现（这段是本讲的精华）

初版讲义我写的是教科书说法：**"CUDA Event 比 `perf_counter` 更准，推荐用 Event"**。
但在你的机器上跑出来是 **Event 6.56 ms vs sync 6.09 ms** —— Event 反而**更大**。这违反直觉，所以我做了三步追查。

### 第 1 步：是方法差异还是时钟漂移？

交替跑 A/B 四轮（先热身让时钟稳定）：

| round | A. sync+perf | B. Event | 差 |
|---|---|---|---|
| 0 | 6.1640 | 6.5737 | +6.6% |
| 1 | 6.0334 | 6.5644 | +8.8% |
| 2 | 6.2538 | 6.5952 | +5.5% |
| 3 | 6.5080 | 7.0805 | +8.8% |

差值**稳定为正**，连 `min` 都差 → 不是随机噪声。

### 第 2 步：差值来自"计时器"还是"有没有间隙"？

设计三个方法：A（sync+perf）、B（Event+每次 sync）、C（Event+背靠背满载）：

| 方法 | median | min |
|---|---|---|
| A. sync + perf_counter | 5.7548 | 5.6581 |
| B. Event + 每次 sync | 6.2064 | 6.1436 |
| C. Event + 背靠背满载 | 6.1464 | 6.1275 |

`A vs B` 差 **7.8%**，`B vs C` 只差 **1.0%**。
→ **差异来自计时器本身，不是满载降频。**

### 第 3 步：差值是固定的还是成比例的？

把每次测量的工作量从 1 个 matmul 加到 20 个：

| 工作量 | A | B | B−A | (B−A)/A |
|---|---|---|---|---|
| 1× | 5.783 | 6.193 | 0.410 | 7.1% |
| 2× | 11.390 | 12.440 | 1.050 | 9.2% |
| 5× | 28.458 | 30.927 | 2.469 | 8.7% |
| 10× | 57.263 | 63.454 | 6.192 | 10.8% |
| 20× | 113.580 | 123.833 | 10.253 | 9.0% |

比例**稳定在 ~9%** → 不是固定开销，是**成比例的系统性偏差**。

### 第 4 步：找一个"不可能错"的裁判

裁判方法：**连续跑 1000 个 matmul 再统一 synchronize**。
因为 CPU 提交只要 14.8 µs，远小于 kernel 的 5.7 ms，CPU 能一直跑在 GPU 前面 → GPU 永不空转 → `总墙钟 / 1000` 必然是单个 kernel 的真实耗时。

```
裁判（1000 个连续 kernel 的平均值）: 5.6886 ms

方法                    median       min      vs 裁判
-------------------------------------------------------
裁判(连续满载)             5.6886    5.6886          —
A. sync+perf           5.7760    5.6761      +1.5%
B. Event+sync          6.2287    6.1106      +9.5%
```

### 结论（这条要记一辈子）

> **在这台机器（WSL2）上，CUDA Event 逐次计时会系统性偏高约 9.5%，而 `perf_counter + synchronize` 与裁判仅差 1.5%（`min` 甚至几乎完全吻合）。**

**所以本项目的标准测量方法是：**

```python
def bench(fn, warmup=10, rep=50):
    """本项目标准计时器：perf_counter + synchronize + 中位数"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median": ts[len(ts) // 2], "min": ts[0], "max": ts[-1]}
```

**三条配套纪律：**

1. **同一份报告里只用一种方法。** A 和 B 差 9%，混用就等于自己制造矛盾。
2. **永远报告方法名 + warmup 次数 + 重复次数。** 例：`6.09 ms (perf_counter+sync, warmup=10, rep=50, median)`。
3. **在原生 Linux 上要重新验证这个结论。** 这是 WSL2 的特性（很可能与 GPU 虚拟化层的 API 延迟有关），不是 CUDA 的普遍规律。等你以后在公司服务器上跑，第一件事就是重跑这个裁判实验。

> **这一段的真正价值不是"Event 不准"，而是这个方法论：**
> **任何测量工具都要用一个独立的、不可能错的裁判去校验。**
> 你后面 13 周会无数次说"我把它优化快了 3 倍" —— 如果测量工具本身有 9% 的偏差，而且你还换了工具，那这些数字全是故事。
>
> 顺带一说：**我原本准备教你的教科书结论是错的。** 发现它的方法就是"多跑一个对照实验"，而不是"相信权威"。

---

## ⑦ 下一讲预告

**L03 · nsys 第一条 timeline**
你写的每个 kernel 在 GPU 上都有一段真实的时间轴：什么时候启动、跑多久、中间有没有空隙。这一讲用 `nsys` 把 timeline 画出来，你会第一次**看见** kernel 之间那些"看不见的空隙"—— CPU 提交不过来、同步等待、隐式拷贝，全都在那儿。
