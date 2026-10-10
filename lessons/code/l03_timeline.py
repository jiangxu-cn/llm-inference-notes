#!/usr/bin/env python3
"""L03 · 看见 kernel 之间的空隙

用法：
    cd ~/work
    ~/work/venv/bin/python l03_timeline.py

导出 l03_trace.json，用浏览器打开 https://ui.perfetto.dev 拖进去即可看时间轴。

────────────────────────────────────────────────────────────────────────
【本文件用到的方法 / 函数速查】

torch.randn(shape, device=)
    生成标准正态分布的随机张量。device="cuda" 表示直接分配在显存里。
    （不写 device 就分配在内存，后面的矩阵乘变成 CPU 运算，测不到 GPU。）

torch.cuda.synchronize()
    阻塞当前 CPU 线程，直到该 GPU 上所有已提交的任务全部执行完毕。
    CUDA 是异步的：不加这句，程序会在 kernel 还没跑完时就往下走。

kernel = lambda: a @ b
    把一个无参表达式包成「无参函数」，等价于 def kernel(): return a @ b。
    这么写是为了下面能反复调用，而不用重复写 a @ b。
    （lambda 里只能放一个表达式，不能放语句。）

torch.profiler.profile(activities=[...])
    上下文管理器：进入 with 块开始采集，退出时停止。
    activities 指定采集哪些「活动」，可以同时给多个：
      ProfilerActivity.CPU  → Python 侧：算子调用、耗时、调用栈
      ProfilerActivity.CUDA → GPU 侧：kernel 名字与耗时（底层走 CUPTI）
    只写 CPU 的话 GPU 泳道会是空的 —— 这是最常见的坑。

prof.export_chrome_trace(path)
    把采集结果导出成 Chrome Trace 格式的 JSON。
    这个格式是通用时间轴格式：chrome://tracing、Perfetto（ui.perfetto.dev）
    以及很多其他工具都能直接打开。

torch.cuda.nvtx.range_push(name) / range_pop()
    在时间轴上开 / 关一个命名区间（NVTX 标记），给自己的代码分段打标签。
    必须成对出现；跨线程 push/pop 会错乱。
    更安全的写法是上下文管理器：with torch.cuda.nvtx.range("名字"): ...

json.load(file)
    读 JSON 文件 → Python 对象（这里是 dict）。

d.get(key)
    安全地取字典的键：键不存在时返回 None，不会抛 KeyError。

sorted(iterable, key=...)
    排序并返回新 list。key 指定「按什么排」，这里按事件起始时间 ts 排。
    括号里的 (e for e in events if ...) 是生成器表达式，只保留满足条件的元素。

e["ts"] / e["dur"]
    事件在时间轴上的起点 / 持续时间，单位都是**微秒**。
    GPU kernel 事件的 cat 字段是 "kernel"，显存拷贝是 "gpu_memcpy"。

────────────────────────────────────────────────────────────────────────
【整体流程】

  1. 造两个 2048×2048 张量，先 warmup 5 次
     （L02 的教训：第一次调用要 200 ms 级，是稳态的 30 倍）

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

import json

import torch
from torch.profiler import ProfilerActivity, profile

N = 20

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")
kernel = lambda: a @ b

for _ in range(5):
    kernel()
torch.cuda.synchronize()

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

prof.export_chrome_trace("l03_trace.json")

events = json.load(open("l03_trace.json"))["traceEvents"]
gpu = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])

print(f"CUDA kernel 事件数: {len(gpu)}")
print(f"kernel 名字示例  : {gpu[0]['name'][:60]}")
print()

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
