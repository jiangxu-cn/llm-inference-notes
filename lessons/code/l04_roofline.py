#!/usr/bin/env python3
"""L04 · 算术强度与 Roofline

回答一个问题：**这个算子到底卡在算力上，还是卡在带宽上？**

用法：
    source ~/work/venv/bin/activate
    cd ~/work
    python l04_roofline.py          # 输出 l04_roofline.png
"""

"""
依赖：
  time        标准库，计时
  torch       张量计算 + CUDA
  plot_style  本项目画图公共设置（中文字体 + 配色）
"""
import time

import matplotlib.pyplot as plt
import torch

import plot_style as ps

"""
硬件参数。Roofline 就是拿这两条天花板去卡所有算子。

PEAK_TFLOPS  FP32 理论峰值算力。RTX 5070 = 6144 个 CUDA 核心 × 2（乘加）× 约 2.5 GHz ≈ 30 TFLOPS
PEAK_BW_GB   显存带宽。12 GB GDDR7 / 192 bit / 28 Gbps ≈ 672 GB/s
RIDGE        屋脊点 = 算力 ÷ 带宽，单位 FLOPs/Byte
             算术强度低于它 → 带宽是瓶颈；高于它 → 算力是瓶颈
"""
PEAK_TFLOPS = 30.0
PEAK_BW_GB = 672.0
RIDGE = PEAK_TFLOPS * 1e12 / (PEAK_BW_GB * 1e9)

"""
标准计时方法（L02 确立）：warmup + 每次 synchronize + 取中位数。
"""
def bench(fn, warmup=10, rep=50):
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
    return ts[len(ts) // 2]


"""
四个被测算子，以及它们的**理论算术强度**。

算术强度 = FLOPs ÷ Bytes
    FLOPs  这个算子要做多少次浮点运算
    Bytes  它**最少**要从显存搬多少字节（只算"必须访问"的那份，
           缓存/分块带来的复用不算在内 —— 那是可实现的下界）

  GEMM（n×n 乘 n×n）
      运算 2n³         每个输出元素做 n 次乘加，乘加各算一次
      搬运 3n² 个 float（读 A + 读 B + 写 C）
      强度 2n³ / (12n²) = n/6      ← **随 n 线性增长**，所以大矩阵一定是算力瓶颈

  GEMV（m×n 矩阵乘 n 维向量 ＝ batch=1 的 decode）
      运算 2mn
      搬运 ≈ 4mn 字节（读矩阵占绝对多数）
      强度 2mn / 4mn = 0.5         ← **与尺寸无关，恒定 0.5**

  Softmax（一行的 N 个元素）
      运算 ≈ 5N（找最大值 N 次比较、减+exp 2N、求和 N、相除 N）
      搬运 8N 字节（读 N + 写 N，每个 float 4 字节）
      强度 ≈ 0.625                 ← 与尺寸无关

  LayerNorm（一行的 N 个元素）
      运算 ≈ 7N（求均值 N、求方差 2N、归一化 2N、缩放平移 2N）
      搬运 8N 字节
      强度 ≈ 0.875                 ← 与尺寸无关

这就是 Roofline 最反直觉的地方：**矩阵乘和归一化的差别不在"算法复杂度"，
而在"每搬一个字节能干多少次活"。**
"""
N = 4096
A = torch.randn(N, N, device="cuda")
B = torch.randn(N, N, device="cuda")
v = torch.randn(N, device="cuda")

OPS = [
    ("GEMM 4096³", 2 * N ** 3, 4 * 3 * N * N, lambda: A @ B),
    ("GEMV (decode)", 2 * N * N, 4 * (N * N + N + N), lambda: A @ v),
    ("Softmax", 5 * N * N, 4 * 2 * N * N, lambda: torch.softmax(A, dim=-1)),
    ("LayerNorm", 7 * N * N, 4 * 2 * N * N, lambda: torch.nn.functional.layer_norm(A, (N,))),
]

"""
逐个测量，算出三个数：
    算术强度   flops / bytes
    实测算力   flops / 耗时
    理论上限   min(峰值算力, 算术强度 × 带宽)   ← Roofline 的答案
"""
print(f"RTX 5070 屋脊点 = {RIDGE:.1f} FLOPs/Byte")
print(f"（算术强度低于 {RIDGE:.1f} 就是带宽瓶颈，高于它就是算力瓶颈）")
print()
print(f"{'算子':<16}{'算术强度':>12}{'实测 TFLOPS':>14}{'理论上限':>12}{'达成率':>10}{'瓶颈':>10}")
print("-" * 76)

results = []
for name, flops, bytes_, fn in OPS:
    ms = bench(fn)
    ai = flops / bytes_
    tf = flops / (ms / 1e3) / 1e12
    ceil = min(PEAK_TFLOPS, ai * PEAK_BW_GB / 1000)
    bound = "算力" if ai > RIDGE else "带宽"
    results.append((name, ai, tf, ceil, bound))
    print(f"{name:<16}{ai:>12.3f}{tf:>14.2f}{ceil:>12.2f}{tf / ceil * 100:>9.1f}%{bound:>10}")

"""
画 Roofline 图。横轴算术强度（对数），纵轴实测算力（对数）。

两条天花板：
    y = AI × 带宽     ← 斜线，带宽限制区（左上被截）
    y = 峰值算力       ← 水平线，算力限制区
两者交点就是屋脊点。每个算子画成一个点 —— **点离哪条天花板近，就说明它被什么卡住**。

ax.loglog()    横纵都取对数。算术强度会跨好几个数量级，必须用对数轴。
ax.fill_between()  给"不可能达到"的区域填色，视觉上一眼看出边界在哪。
"""
ps.setup_cjk_font()

fig, ax = plt.subplots(figsize=(11, 7.5))

ai_axis = [0.1, 1000]
mem_line = [ai * PEAK_BW_GB / 1000 for ai in ai_axis]
ax.loglog(ai_axis, mem_line, "--", color=ps.CEIL_COLOR, lw=2, label=f"带宽上限（{PEAK_BW_GB:.0f} GB/s）")
ax.axhline(PEAK_TFLOPS, ls="--", color=ps.CEIL_COLOR, lw=2, label=f"算力上限（{PEAK_TFLOPS:.0f} TFLOPS）")
ax.axvline(RIDGE, ls=":", color=ps.GAP_COLOR, lw=1.5)
ax.fill_between(ai_axis, [PEAK_TFLOPS] * 2, [PEAK_TFLOPS * 3] * 2, color=ps.GAP_COLOR, alpha=0.08)
ax.annotate(f"屋脊点 {RIDGE:.1f}", xy=(RIDGE, 2), xytext=(RIDGE * 1.15, 2.2),
            color=ps.GAP_COLOR, fontsize=11)

"""
每个点的标签位置单独给（相对该点的倍数）。
用同一套偏移的话，左下角那三个带宽瓶颈的算子会挤在一起看不清。
顺序要和 OPS 一致：GEMM / GEMV / Softmax / LayerNorm。
"""
LABEL_POS = [(0.28, 1.9), (0.26, 2.9), (2.7, 0.85), (1.05, 2.7)]

for (name, ai, tf, ceil, bound), (dx, dy) in zip(results, LABEL_POS):
    ax.plot(ai, tf, "o", ms=11, color=ps.POINT_COLOR, zorder=5)
    ax.annotate(f"{name}\n达成 {tf / ceil * 100:.0f}%",
                xy=(ai, tf), xytext=(ai * dx, tf * dy),
                fontsize=10, color=ps.POINT_COLOR,
                arrowprops=dict(arrowstyle="-", color=ps.POINT_COLOR, lw=0.8, alpha=0.6))

ax.set_xlabel("算术强度（FLOPs / Byte）—— 每搬 1 字节能做多少次浮点运算", fontsize=12)
ax.set_ylabel("实测算力（TFLOPS，对数轴）", fontsize=12)
ax.set_title("RTX 5070 Roofline：四个常见算子卡在哪条天花板上", fontsize=14)
ax.set_xlim(0.2, 2000)
ax.set_ylim(0.01, 60)
ax.grid(which="both", alpha=0.25)
ax.legend(loc="lower right", fontsize=11)

fig.tight_layout()
fig.savefig("l04_roofline.png", dpi=150)
print()
print("已生成 l04_roofline.png")
