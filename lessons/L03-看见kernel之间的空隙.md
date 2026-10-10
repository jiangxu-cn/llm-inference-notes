# 第 3 讲 · 看见 kernel 之间的空隙

> W1 / L03 · 预计 30 min 理解 + 60–90 min 实操
> 通过标准：能用 chrome trace 打开时间轴 + 说清"每次 sync 在 GPU 上凿了多大的洞"

---

## ① 依赖门（1 min）

**✅ 现在只需要补这些：**
1. 知道 kernel 在 GPU 上"排队执行"（L02 已建立）
2. 会看一张"横轴是时间、纵轴是设备"的甘特图
3. 会拖文件到浏览器

**⏸️ 现在不学也没影响：**
CUPTI 原理、GPU 硬件队列（Hyper-Q）、驱动调度器、nsys 的 SQLite 后端 —— 全部后面才用。

**🔜 什么时候会回来：**
- **第 8 讲（W8）** 专门做 profiling：会用 `ncu` 看单个 kernel 的 SOL
- **第 12 讲（W12）** 会用同一套方法看 vLLM 的 prefill/decode 交替

---

## ② 直觉先行（6 min）

L02 你学到：**CPU 是下单的，GPU 是做菜的**。但那是"单个菜"的视角。

现在是"一整晚的生意"：

```
时间 ──────────────────────────────────────────────▶

CPU ║下单║…║下单║…║下单║…║下单║…║下单║…║下单║…
      ↓      ↓      ↓      ↓      ↓      ↓
GPU ║███菜1███║███菜2███║███菜3███║███菜4███║███菜5███║
              ↑↑↑        ↑↑↑        ↑↑↑
            空隙        空隙        空隙
```

**GPU 每一段"什么都没干"的时间，就是空隙（gap）。**

空隙从哪来？
- CPU 提交下一道菜准备得太慢（Python 开销、launch 延迟）
- 你按了 `synchronize()` —— **厨房必须停下来等你确认**
- 隐式的显存拷贝、内存分配

**本讲要回答的问题：空隙有多大？占总时间多少？**

---

## ③ 原理（8 min）

### 3.1 三种 profiling 工具，各看一层

| 工具 | 看什么 | 粒度 | 你的机器上 |
|---|---|---|---|
| **`torch.profiler`** | CPU 调用栈 + GPU kernel 时间轴 | 中观 | ✅ **本讲用它** |
| **`nsys`** | 系统级 timeline（CPU/GPU/OS/网络一起） | 宏观 | ⚠️ 见附录（GPU 侧抓不到） |
| **`ncu`** | **单个 kernel 内部**：SOL、occupancy、bank conflict | 微观 | ✅ 但需先开权限 |

**别混用**：想回答"为什么整体慢"用 timeline；想回答"这个 kernel 为什么慢"用 ncu。

### 3.2 时间轴上有三条线

```
Thread (CPU)   ║  ← Python 代码、cuLaunchKernel、cudaDeviceSynchronize
      ↓ 提交
Stream (GPU)   ║  ← kernel 真正在跑
      ↓
Memory         ║  ← H2D / D2H 拷贝
```

**优化 GPU 利用率 = 让 Stream 那条线尽量没有空白。**

### 3.3 本机实测：sync 到底凿了多大的洞？

同一批 kernel（2048³ FP32 matmul，20 个），两种提交方式：

| | A. 背靠背提交 | B. 每次 `synchronize()` |
|---|---|---|
| 时间轴跨度（第 1 次跑 / 第 2 次跑） | 14.06 / 14.32 ms | 16.04 / 17.40 ms |
| kernel 累计时长 | 13.88 / 14.26 ms | 14.14 / 14.48 ms |
| **GPU 忙碌占比** | **98.7% / 99.6%** | **88.1% / 83.2%** |
| **平均空隙** | **9.6 / 2.9 µs** | **100.1 / 153.7 µs** |
| 最大空隙 | 127.9 / 4.5 µs | 178.3 / 272.1 µs |

> ⚠️ **两次跑的绝对数字不一样**（A 的平均空隙 9.6 → 2.9 µs，B 的 100 → 154 µs）。这是正常的测量浮动。
> **但结论完全稳定：差距在 10–50 倍量级。**
>
> **记住结论，不要记小数位** —— 这又是第 2 讲那条纪律：报告时永远带上"哪次跑、什么条件"。

**结论：每次 `synchronize()` 在 GPU 上凿出 ~100–150 µs 的洞 —— 比背靠背提交大一到两个数量级。**

20 次就是 **2–3 ms 的纯浪费**，占总时间的 12–17%。

> 💡 **这就是为什么"循环里不要随便 synchronize"**。L02 的 `bench()` 每次都同步，是**故意的**（要隔离测量）；但在生产代码里，每次迭代都同步 = 白送 12% 的性能。

---

## ④ 实操（12 min）

代码：[`code/l03_timeline.py`](code/l03_timeline.py)（约 80 行）

```bash
cd ~/work
~/work/venv/bin/python l03_timeline.py
```

**核心代码：**

```python
import json
import torch
from torch.profiler import ProfilerActivity, profile

N = 20
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")

def kernel():
    return a @ b

for _ in range(5):                      # warmup（L02 的教训）
    kernel()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # ---- 阶段 A：背靠背提交，最后统一等待 ----
    torch.cuda.nvtx.range_push("A_burst")
    for _ in range(N):
        kernel()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    # ---- 阶段 B：每提交 1 个就等一次 ----
    torch.cuda.nvtx.range_push("B_sync_each")
    for _ in range(N):
        kernel()
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

prof.export_chrome_trace("l03_trace.json")      # ← 时间轴导出

# ============ 从 trace 里算 GPU 空隙 ============
events = json.load(open("l03_trace.json"))["traceEvents"]
gpu = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])

for label, ks in {"A 背靠背提交": gpu[:N], "B 每次 sync": gpu[N:2*N]}.items():
    gaps = [ks[i]["ts"] - (ks[i-1]["ts"] + ks[i-1]["dur"]) for i in range(1, len(ks))]
    span = (ks[-1]["ts"] + ks[-1]["dur"]) - ks[0]["ts"]
    busy = sum(k["dur"] for k in ks)
    print(f"【{label}】busy={busy/span*100:.1f}%  avg_gap={sum(gaps)/len(gaps):.1f}us")
```

**预期输出（本机实跑）：**

```
CUDA kernel 事件数: 40
kernel 名字示例  : void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_8x4_nn_a

【A 背靠背提交】20 个 kernel
    跨度        :   14.058 ms
    kernel 总时长:   13.877 ms
    GPU 忙碌占比 :    98.7%
    平均空隙    :      9.6 us
    最大空隙    :    127.9 us

【B 每次 sync】20 个 kernel
    跨度        :   16.043 ms
    kernel 总时长:   14.141 ms
    GPU 忙碌占比 :    88.1%
    平均空隙    :    100.1 us
    最大空隙    :    178.3 us

chrome trace 已导出: l03_trace.json
```

### 看真正的时间轴（推荐做一次）

1. 浏览器打开 **https://ui.perfetto.dev**（在线，不用装东西）
2. 把 [`code/traces/l03_trace.json`](code/traces/l03_trace.json) 拖进去
3. 你会看到两条"泳道"：
   - 上面是 **CPU 线程**（一堆 `aten::mm`、`cuLaunchKernel`）
   - 下面是 **GPU Stream**（一串绿色的 kernel 块）

**放大看阶段 B** —— 你会亲眼看到 kernel 之间那些明显的空白。阶段 A 的 kernel 是紧贴的。

---

## ⑤ 语法与坑点（2 min）

| 语法 / 坑 | 说明 |
|---|---|
| `ProfilerActivity.CUDA` **必须显式加** | 只写 `CPU` → GPU 泳道是空的，你会以为"GPU 没干活" |
| `e["cat"] == "kernel"` | trace 里 GPU kernel 的类别是 `kernel`；`gpu_memcpy` / `gpu_memset` 是另外的类别 |
| `ts` 和 `dur` 单位是 **微秒** | 别当成毫秒 |
| `torch.cuda.nvtx.range_push/pop` **必须成对** | 忘了 pop → 区间永远不闭合；跨线程 push/pop 会导致错乱。安全写法用 `with torch.cuda.nvtx.range("name"):` |
| `prof.export_chrome_trace()` 要在 `with` 块**之后**调用 | 在块内调用只能拿到部分数据 |
| `with_stack=True` / `record_shapes=True` 会显著拖慢 | 只在需要时开，做性能测量时**关掉** |
| Profiler 本身有开销 | 测出来的"空隙"里混着 profiler 自己的开销。所以**要用两个独立方法交叉验证**（下面附录就是） |

---

## ⑥ 自测 3 题（1 min）

1. 阶段 A 和 B 的 **kernel 累计时长**几乎一样（13.877 vs 14.141 ms），但 **GPU 时间轴跨度**差了 2 ms。**这 2 ms 去哪了？** 说明什么？
2. 你有一段推理代码：`for token in range(1000): logits = model(x); x = sample(logits); print(x)`。**猜测 timeline 上会看到什么？** 怎么改？
3. 你的 kernel 累计 10 ms，但墙钟跑了 25 ms。**列出至少 3 个可能吃掉那 15 ms 的东西。**

---

## 🔬 附录：nsys 在这台机器上的静默失败（这段很重要）

按讲义原计划，L03 应该用 `nsys` 画 timeline。**但在你的机器上它失败了，而且是静默失败。**

### 现象

`nsys profile` 正常跑完、正常生成 334 KB 的 `.nsys-rep`、**没有任何报错**。但一提取报告：

```
$ nsys stats --report cuda_gpu_kern_sum l03.nsys-rep
SKIPPED: l03.sqlite does not contain CUDA kernel data.

$ nsys stats --report cuda_gpu_mem_time_sum l03.nsys-rep
SKIPPED: l03.sqlite does not contain GPU memory data.
```

逐项检查报告有没有数据：

| 报告 | 数据 |
|---|---|
| `cuda_api_sum`（CPU 侧的 CUDA API 调用） | ✅ 19 行 |
| `nvtx_sum`（NVTX 区间） | ✅ 4 行 |
| `cuda_gpu_kern_sum`（**GPU kernel**） | ❌ **空** |
| `cuda_gpu_mem_time_sum`（**显存拷贝**） | ❌ **空** |

**NVTX 区间数据本身还挺有用**（这也是我们最后拿到的唯一 nsys 数据）：

| NVTX 区间 | 总时长 |
|---|---|
| `C_pageable_H2D`（5 次从可分页内存拷 128MB） | **63.50 ms** |
| `D_pinned_H2D`（5 次从锁页内存拷 128 MB） | **24.41 ms** |
| `B_sync_each`（20 kernel，每次 sync） | 15.87 ms |
| `A_burst_20kernels`（20 kernel，只 sync 一次） | **1.01 ms** |

> 顺带得到一个经典结论：**锁页内存（pinned）比可分页内存快 2.6 倍**（24.41 vs 63.50 ms，同样的 640 MB）。因为可分页内存的 DMA 需要先拷到一块临时锁页缓冲区。

### 追查：是 CUPTI 坏了吗？

不是。用 `torch.profiler` 一测，**CUDA kernel 数据完全正常**：

```
void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...>   Self CUDA 7.362ms   736.166us/call   10 calls
```

**所以 CUPTI 的 GPU activity 通道是通的 —— 问题出在 nsys 自己。**

### 最可能的原因

```
nsys 版本 : 2024.6.2.225（2024-12-11 构建）
GPU 驱动  : 617.42（2026-09-30，CUDA UMD 13.4）
```

**nsys 太老了。** 它自带的是 `libcupti.so.12.8`（2024 年的），而驱动已经是 CUDA 13.4 时代。CUPTI 要求与驱动版本兼容 —— API 拦截（nsys 自己的注入库做的）还能工作，但 GPU activity 追踪静默失效。

### 修复方案（可选，一条命令）

装一个更新的 nsys（`wsl-ubuntu` 源里有 CUDA 13-x 系列的 nsys）：

```bash
sudo apt-get install -y cuda-nsight-systems-13-3
```

> 这条**不是必须的** —— `torch.profiler` 的 chrome trace 已经覆盖了 L03 到 L12 的全部 timeline 需求，而且比 nsys 更简单。等 W8 做完 `ncu` 之后，如果你想升级再来做。

### ⚠️ 真正的教训

> **`nsys` 抓不到 GPU 数据时，它不会报错，只会给你一个"成功"的空报告。**
>
> 如果你不检查报告里到底有没有数据，就会得到"GPU 一直很忙、没有空隙"这种**看起来合理但完全错误**的结论。
>
> **这和第 2 讲的教训是同一条：先验证你的测量工具，再去相信它的数字。**
>
> 具体到操作上：**每次生成报告后，先看一眼行数/条数是不是 0。**

---

## 🔗 顺带把 L02 的谜团解开了

L02 我们发现 `bench()`（每次 sync）比"连续 1000 kernel 的裁判"高 **1.5%**（5.776 vs 5.689 ms），当时只知道是测量方法差异。

现在有了 L03 的数据，答案很清楚了：

```
bench() 每次迭代做的是：
    t0 = perf_counter()
    fn()            ← Python 派发 + launch，约 100 µs，这期间 GPU 是空的
    synchronize()   ← 等 kernel 跑完
    t1 = perf_counter()

所以 bench() 测的是「派发时间 + kernel 时间」
      裁判测的是     「纯 kernel 时间」
差值 = 派发时间 ≈ 100 µs
```

核对：`100 µs / 5689 µs = 1.76%`，与 L02 实测的 `+1.5%` 吻合。

**两个独立的实验互相印证 —— 这才是「可信的测量」。**

---

## ⑦ 下一讲预告

**L04 · 算术强度与 Roofline**
到这里你已经会测了（L02）、会看时间轴了（L03）。下一讲回答一个更根本的问题：
**这个 kernel 到底还有多少优化空间？**
答案是算一个数：**算术强度 = FLOPs ÷ Bytes**。它决定了你的 kernel 是卡在算力上还是带宽上 —— 而这两种情况的优化手段**完全相反**。
这是整个 W2 的核心，也是面试里最能分辨真假的一道题。
