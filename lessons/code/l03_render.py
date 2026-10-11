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
from matplotlib import font_manager as fm
from matplotlib.patches import Patch

KERNEL_COLOR = "#3f9e63"
GAP_COLOR = "#d94f4f"
N = 20

"""
设置中文字体。不设的话，图里所有汉字都会变成方框（俗称「豆腐块」）。

font_manager.fontManager.addfont(path)
    把一个字体文件注册进 matplotlib 的字体表，之后就能按名字引用它。

FontProperties(fname=path).get_name()
    从字体文件里读出它真正的字体名（比如 "Microsoft YaHei"），
    这个名字才是要填进 font.family 的东西。

plt.rcParams["font.family"]
    全局默认字体族。

plt.rcParams["axes.unicode_minus"] = False
    中文字体里的「减号」是另一个码位，不关掉的话坐标轴上的负数会变成方框。
    这是中文 matplotlib 最经典的一个坑。

字体从 Windows 目录取（WSL 直接能读，且不用 sudo 装）：
微软雅黑 -> 黑体 -> 宋体，谁先加载成功就用谁。
"""
CJK_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "/mnt/c/Windows/Fonts/simhei.ttf",    # 黑体
    "/mnt/c/Windows/Fonts/simsun.ttc",    # 宋体
]


def setup_cjk_font():
    for path in CJK_CANDIDATES:
        try:
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    return None


"""
读 trace 并拆成两个阶段。

json.load(open(path))["traceEvents"]
    读 JSON → dict → 取出事件列表（list of dict）。

sorted(..., key=lambda e: e["ts"])
    按事件起始时间排序，保证 kernel 是按时间顺序排的。

kernels[:N] / kernels[N:2*N]
    前 20 个是阶段 A，后 20 个是阶段 B。

t0 = kernels[0]["ts"]
    所有时间都相对第一个 kernel 的起点表示，不然 x 轴会是一串天文数字。
"""
events = json.load(open("l03_trace.json"))["traceEvents"]
kernels = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
phase_a, phase_b = kernels[:N], kernels[N:2 * N]
t0 = kernels[0]["ts"]


"""
把原始事件转成画图要用的「横段」格式。

burst(ks) -> [(起点, 宽度), ...]
    单位从微秒换成**毫秒**（x 轴用毫秒好读）。

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
拼一行中文摘要给图当标题用。

注意单位：gaps() 返回的宽度是**毫秒**（为了画 x 轴方便），
但空隙用毫秒不好读 —— 0.147 ms 不如 147 µs 直观。
所以这里乘 1000 换回微秒再打印。
"""
def summary(ks, label):
    g = [w * 1000 for _, w in gaps(ks)]
    span = ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]
    busy = sum(k["dur"] for k in ks)
    return f"{label}　忙碌 {busy / span * 100:.1f}%　平均空隙 {sum(g) / len(g):.1f}µs　最大空隙 {max(g):.1f}µs"


"""
画三个子图。

plt.subplots(3, 1, figsize=(宽, 高))
    一次创建 3 行 1 列的子图，返回 (fig, axes)。

ax.broken_barh(横段列表, (y起点, 高度))
    画一组水平色块 —— 最适合甘特图 / 时间轴。

ax.set_xlim / set_yticks / ax.set_yticklabels
    控制坐标轴显示范围和刻度标签。

ax.legend(handles=[Patch(...)])
    Patch 是「图例色块」对象，用来手动造图例（因为 broken_barh 不会自动生成图例项）。

fig.tight_layout()   自动调整子图间距，防止标题被裁掉。
fig.savefig(path, dpi=150)   存 PNG，dpi 越高越清晰、文件越大。
"""
used_font = setup_cjk_font()
print(f"使用中文字体: {used_font or '未找到（中文会显示成方框）'}")

fig, axs = plt.subplots(3, 1, figsize=(13, 9))

ax = axs[0]
ax.broken_barh(burst(phase_a), (0.60, 0.32), color=KERNEL_COLOR)
ax.broken_barh(burst(phase_b), (0.08, 0.32), color=KERNEL_COLOR)
ax.broken_barh(gaps(phase_b), (0.08, 0.32), color=GAP_COLOR, alpha=0.5)
ax.set_ylim(0, 1)
ax.set_yticks([0.76, 0.24])
ax.set_yticklabels(["A 背靠背提交", "B 每次 sync"])
ax.set_title("完整时间轴 —— 每阶段 20 个 kernel（红色 = GPU 空闲的空隙）")
ax.grid(axis="x", alpha=0.3)
ax.legend(
    handles=[Patch(color=KERNEL_COLOR, label="kernel 执行中"),
             Patch(color=GAP_COLOR, label="GPU 空闲（空隙）")],
    loc="upper right", fontsize=10,
)

for ax, ks, label in [(axs[1], phase_a, "A 背靠背提交（只在最后 sync 一次）"),
                      (axs[2], phase_b, "B 每次迭代都 synchronize()")]:
    z = ks[:5]
    ax.broken_barh(burst(z), (0.35, 0.35), color=KERNEL_COLOR)
    ax.broken_barh(gaps(z), (0.35, 0.35), color=GAP_COLOR, alpha=0.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim((z[0]["ts"] - t0) / 1000, (z[-1]["ts"] + z[-1]["dur"] - t0) / 1000)
    ax.set_title(summary(ks, label) + "　［放大：前 5 个 kernel］")
    ax.set_xlabel("距第一个 kernel 的时间（毫秒）")

fig.tight_layout()
fig.savefig("l03_timeline.png", dpi=150)
print("已生成 l03_timeline.png")
