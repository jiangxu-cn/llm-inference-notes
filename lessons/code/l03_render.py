#!/usr/bin/env python3
"""L03 · 把 chrome trace 画成图片（完全离线，不需要任何网站）

用法：
    source ~/work/venv/bin/activate
    cd ~/work
    python l03_render.py             # 读 l03_trace.json，输出 l03_timeline.png
"""

"""
依赖：
  json        Python 标准库，读 trace 文件
  matplotlib  画图库
"""
import json

import matplotlib

"""
matplotlib.use("Agg") 必须写在 import pyplot **之前**。
它把绘图后端切成「只输出文件、不开窗口」。
WSL 里没有图形界面，不切后端的话 import pyplot 就会报错。
"""
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KERNEL_COLOR = "#3f9e63"
GAP_COLOR = "#d94f4f"
N = 20

"""
读 trace 并拆成两个阶段。

json.load(open(path))["traceEvents"]
    读 JSON → dict → 取出事件列表（list of dict）。

sorted(..., key=lambda e: e["ts"])
    按事件起始时间排序，保证内核是按时间顺序排的。

kernels[:N] / kernels[N:2*N]
    前 20 个是阶段 A，后 20 个是阶段 B。

t0 = kernels[0]["ts"]
    把所有时间都相对第一个 kernel 的起点来表示，不然 x 轴会是一串天文数字。
"""
events = json.load(open("l03_trace.json"))["traceEvents"]
kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
phase_a, phase_b = kernels[:N], kernels[N:2 * N]
t0 = kernels[0]["ts"]


"""
把原始事件转成画图要用的「横段」格式。

burst(ks) -> [(起点, 宽度), ...]
    单位从微秒换成毫秒。

gaps(ks)  -> [(起点, 宽度), ...]
    第 i 个空隙的起点 = 前一个 kernel 的结束时刻
    宽度            = 下一个 kernel 的起点 − 前一个的结束时刻
"""
def burst(ks):
    return [((k["ts"] - t0) / 1000, k["dur"] / 1000) for k in ks]


def gaps(ks):
    out = []
    for i in range(1, len(ks)):
        end = ks[i - 1]["ts"] + ks[i - 1]["dur"]
        out.append(((end - t0) / 1000, (ks[i]["ts"] - end) / 1000))
    return out


"""
算三个指标，拼成一行标题文字。逻辑和 l03_timeline.py 里完全一样，
只是这里输出成字符串给图用。

注意单位：gaps() 返回的宽度是**毫秒**（为了画图方便），
但空隙用毫秒不好读 —— 0.147 ms 不如 147 us 直观。
所以这里乘 1000 换回微秒再打印。
"""
def summary(ks, label):
    g = [w * 1000 for _, w in gaps(ks)]
    span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
    busy = sum(k["dur"] for k in ks)
    return f"{label}   busy {busy / span * 100:.1f}%   avg gap {sum(g) / len(g):.1f}us   max gap {max(g):.1f}us"


"""
画三个子图。

plt.subplots(3, 1, figsize=(宽, 高))
    一次创建 3 行 1 列的子图，返回 (fig, axes)；axes 是可下标的数组。

ax.broken_barh(横段列表, (y起点, 高度))
    画一组水平色块 —— 最适合甘特图 / 时间轴。

ax.set_xlim / set_yticks / set_yticklabels
    控制坐标轴显示范围和刻度标签。

fig.tight_layout()   自动调整子图间距，防止标题被裁掉。
fig.savefig(path, dpi=150)   存 PNG，dpi 越高越清晰、文件越大。
"""
fig, axs = plt.subplots(3, 1, figsize=(13, 8.5))

ax = axs[0]
ax.broken_barh(burst(phase_a), (0.6, 0.32), color=KERNEL_COLOR)
ax.broken_barh(burst(phase_b), (0.08, 0.32), color=KERNEL_COLOR)
ax.broken_barh(gaps(phase_b), (0.08, 0.32), color=GAP_COLOR, alpha=0.5)
ax.set_ylim(0, 1)
ax.set_yticks([0.76, 0.24])
ax.set_yticklabels(["A  burst", "B  sync each"])
ax.set_title("Full timeline — 20 kernels per phase (red = GPU idle gaps)")
ax.grid(axis="x", alpha=0.3)

for ax, ks, label in [(axs[1], phase_a, "A  burst (1 sync at the end)"),
                      (axs[2], phase_b, "B  synchronize() every iteration")]:
    z = ks[:5]
    ax.broken_barh(burst(z), (0.35, 0.35), color=KERNEL_COLOR)
    ax.broken_barh(gaps(z), (0.35, 0.35), color=GAP_COLOR, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim((z[0]["ts"] - t0) / 1000, (z[-1]["ts"] + z[-1]["dur"] - t0) / 1000)
    ax.set_title(summary(ks, label) + "     [zoom: first 5 kernels]")
    ax.set_xlabel("time from first kernel (ms)")

fig.tight_layout()
fig.savefig("l03_timeline.png", dpi=150)
print("已生成 l03_timeline.png")
