#!/usr/bin/env python3
"""L03 · 看见 kernel 之间的空隙

用法：
    cd ~/work
    ~/work/venv/bin/python l03_timeline.py

导出 l03_trace.json，用浏览器打开 https://ui.perfetto.dev 拖进去即可看时间轴。
────────────────────────────────────────────────────────────────────────
【整体流程】

  1. 造两个 2048×2048 张量，先 warmup 5 次
     （L02 的教训：第一次调用要 170 ms 级，是稳态的 30 倍）

  2. 在 profiler 里跑两个对照阶段，各 20 个 kernel：
       阶段 A  背靠背提交，最后只 sync 一次   → GPU 应该一直是满的
       阶段 B  每提交一个就 sync 一次          → GPU 中间应该有空隙

  3. 导出 chrome trace

  4. 从 trace 里筛出 GPU kernel 事件并按时间排序
     前 20 个属于阶段 A，后 20 个属于阶段 B

  5. 对每个阶段算三个数：
       忙碌占比 = kernel 时长之和 ÷ 时间轴跨度
       平均空隙 = 相邻两个 kernel 之间 GPU 空闲时间的均值
       最大空隙 = 同上取最大
────────────────────────────────────────────────────────────────────────
"""

"""
依赖两个东西：

  json   Python 标准库。用来读 profiler 导出的 JSON 文件
  torch  张量计算 + CUDA 接口 + 它自带的 profiler
         （torch.profiler 是 torch 的一部分，不用另装）
"""
import json

import torch
from torch.profiler import ProfilerActivity, profile


"""
准备被测对象。和 L02 思路一样：先建好张量，再包一个无参函数反复调用。

torch.randn(2048, 2048, device="cuda")
    生成 2048×2048 标准正态随机张量，直接放显存。
    这里用 2048 而不是 L02 的 4096：kernel 要足够短，
    才能在 20 个 kernel 的尺度上把「空隙」凸显出来。

lambda: a @ b
    包成无参函数，方便下面循环里反复调用。
"""
N = 20

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
kernel = lambda: a @ b


"""
热身 5 次。目的是把初始化开销（kernel 首次加载、cuBLAS handle 创建等）
排除在测量之外，否则阶段 A 会被第一次那 170 ms 的调用污染。
"""
for _ in range(5):
    kernel()
torch.cuda.synchronize()


"""
采集两个对照阶段。这是本讲的核心。

torch.profiler.profile(activities=[...])
    上下文管理器：进入 with 块开始采集，退出时停止。
    activities 指定采集哪些「活动」，可以同时给多个：
      ProfilerActivity.CPU  → Python 侧：算子调用、耗时、调用栈
      ProfilerActivity.CUDA → GPU 侧：kernel 名字与耗时（底层走 CUPTI）
    只写 CPU 的话 GPU 泳道会是空的 —— 这是最常见的坑。

torch.cuda.nvtx.range_push(name) / range_pop()
    在时间轴上开 / 关一个命名区间（NVTX 标记），给自己的代码分段打标签。
    必须成对出现；跨线程 push/pop 会错乱。
    更安全的写法是上下文管理器：with torch.cuda.nvtx.range("名字"): ...

两个阶段的唯一区别：
    A 里 sync 在循环外（提交完 20 个统一等一次）
    B 里 sync 在循环内（每提交一个就等它跑完）
    其余代码完全相同 —— 这就是控制变量。
"""
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    torch.cuda.nvtx.range_push("A_burst")
    for _ in range(N):
        kernel()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push("B_sync_each")
    for _ in range(N):
        kernel()
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()


"""
导出采集结果，然后自己从里面把「空隙」算出来。

prof.export_chrome_trace(path)
    把采集结果导出成 Chrome Trace 格式的 JSON。
    这个格式是通用时间轴格式：chrome://tracing、Perfetto（ui.perfetto.dev）
    以及很多其他工具都能直接打开。

json.load(file)["traceEvents"]
    读 JSON → dict，再取出事件列表（一个 list，每个元素是一个事件 dict）。

e.get("cat")
    安全取字典的键：键不存在时返回 None，不会抛 KeyError。
    GPU kernel 事件的 cat 字段是 "kernel"；显存拷贝是 "gpu_memcpy"。

sorted(iterable, key=...)
    排序并返回新 list。key 指定「按什么排」，这里按事件起始时间 ts 排。
    括号里的 (e for e in events if ...) 是生成器表达式，只保留满足条件的元素。

e["ts"] / e["dur"]
    事件在时间轴上的起点 / 持续时间，单位都是**微秒**（不是毫秒）。

gpu[:N] / gpu[N:2*N]
    list 切片。前 20 个是阶段 A 的 kernel，接下来 20 个是阶段 B 的。
    之所以能这样切，是因为两个阶段用的是同一个 kernel、而且 A 先跑。
"""
prof.export_chrome_trace("l03_trace.json")

events = json.load(open("l03_trace.json"))["traceEvents"]
gpu = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])

print(f"CUDA kernel 事件数: {len(gpu)}")
print(f"kernel 名字示例  : {gpu[0]['name'][:60]}")
print()


"""
对每个阶段统计三个指标并打印。

ends[i] = 第 i 个 kernel 的结束时刻 = 起点 ts + 时长 dur
gaps[i] = 第 i 个 kernel 的起点 − 前一个 kernel 的结束时刻
          → 这段时间 GPU 是闲着的，就是我们要找的「空隙」

span = 整个阶段的时间轴跨度（第一个 kernel 开始 → 最后一个 kernel 结束）
busy = 所有 kernel 时长之和
busy / span = GPU 忙碌占比，剩下那部分就是被空隙吃掉的

f"{x:8.3f}"
    f-string 格式说明符：总宽 8、小数点后 3 位、右对齐。
    对齐是为了让两次运行的数字能竖着比。
"""
for label, ks in {"A 背靠背提交": gpu[:N], "B 每次 sync": gpu[N:2 * N]}.items():
    ends = [k["ts"] + k["dur"] for k in ks]
    gaps = [ks[i]["ts"] - ends[i - 1] for i in range(1, len(ks))]
    span = ends[-1] - ks[0]["ts"]
    busy = sum(k["dur"] for k in ks)
    print(f"【{label}】{len(ks)} 个 kernel")
    print(f"    时间轴跨度   : {span / 1000:8.3f} ms")
    print(f"    kernel 总时长: {busy / 1000:8.3f} ms")
    print(f"    GPU 忙碌占比 : {busy / span * 100:7.1f}%")
    print(f"    平均空隙     : {sum(gaps) / len(gaps):8.1f} us")
    print(f"    最大空隙     : {max(gaps):8.1f} us")
    print()
