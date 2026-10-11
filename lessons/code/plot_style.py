#!/usr/bin/env python3
"""画图公共设置：中文字体 + 统一配色。L03 / L04 及以后所有图共用。

用法：
    import plot_style as ps
    ps.setup_cjk_font()
"""

"""
依赖：
  matplotlib  画图库
"""
import matplotlib

"""
matplotlib.use("Agg") 必须写在 import pyplot **之前**。
切成「只输出文件、不开窗口」的后端 —— WSL 里没有图形界面。
"""
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# 统一配色
KERNEL_COLOR = "#3f9e63"   # 绿：正常执行 / 达标
GAP_COLOR = "#d94f4f"      # 红：空闲 / 瓶颈
POINT_COLOR = "#2f6fd0"    # 蓝：数据点
CEIL_COLOR = "#888888"     # 灰：理论上限线

"""
中文字体候选。matplotlib 默认字体没有汉字，不设的话中文会渲染成方框。

从 Windows 目录取 —— WSL 直接能读，不用 sudo 装任何东西。
按顺序尝试，谁先加载成功用谁。
"""
CJK_CANDIDATES = [
    "/mnt/c/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "/mnt/c/Windows/Fonts/simhei.ttf",    # 黑体
    "/mnt/c/Windows/Fonts/simsun.ttc",    # 宋体
]


"""
注册并启用中文字体，返回实际用上的字体名（找不到就返回 None）。

font_manager.fontManager.addfont(path)
    把字体文件注册进 matplotlib 的字体表。

FontProperties(fname=path).get_name()
    从字体文件里读出真正的字体名（如 "Microsoft YaHei"），
    这个名字才是要填进 font.family 的东西。

rcParams["axes.unicode_minus"] = False
    中文字体里的减号是另一个码位，不关掉的话坐标轴上的负数会变方框。
    这是中文 matplotlib 最经典的坑。
"""
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
