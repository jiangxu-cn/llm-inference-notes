#!/usr/bin/env python3
"""L03 · 看见 kernel 之间的空隙

用 torch.profiler 采集，导出 chrome trace，再自己算 GPU 空闲占比。
导出文件可以用 https://ui.perfetto.dev 打开看时间轴（不需要装任何东西）。

用法：
    ~/work/venv/bin/python l03_timeline.py
"""
import json
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

N = 20
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")


def kernel():
    return a @ b


for _ in range(5):                      # warmup
    kernel()
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # ---- 阶段 A：背靠背提交 20 个 kernel，最后统一等待 ----
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

TRACE = "l03_trace.json"
prof.export_chrome_trace(TRACE)

# ================= 从 trace 里算 GPU 空隙 =================
with open(TRACE) as f:
    events = json.load(f)["traceEvents"]

gpu = sorted((e for e in events if e.get("cat") == "kernel"), key=lambda e: e["ts"])
print(f"CUDA kernel 事件数: {len(gpu)}")
print(f"kernel 名字示例  : {gpu[0]['name'][:60]}")
print()

# 前 N 个是阶段 A，后 N 个是阶段 B
phases = {"A 背靠背提交": gpu[:N], "B 每次 sync": gpu[N:2 * N]}

for label, ks in phases.items():
    gaps = []
    for i in range(1, len(ks)):
        prev_end = ks[i - 1]["ts"] + ks[i - 1]["dur"]
        gaps.append(ks[i]["ts"] - prev_end)
    span = (ks[-1]["ts"] + ks[-1]["dur"]) - ks[0]["ts"]
    busy = sum(k["dur"] for k in ks)
    print(f"【{label}】{len(ks)} 个 kernel")
    print(f"    跨度        : {span/1000:8.3f} ms")
    print(f"    kernel 总时长: {busy/1000:8.3f} ms")
    print(f"    GPU 忙碌占比 : {busy/span*100:7.1f}%")
    print(f"    平均空隙    : {sum(gaps)/len(gaps):8.1f} us")
    print(f"    最大空隙    : {max(gaps):8.1f} us")
    print()

# 反推 CUDA Event 偏差：阶段 B 的墙钟 vs kernel 总时长
ks = phases["B 每次 sync"]
wall_b = max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)
print(f"阶段 B 的 GPU 时间轴跨度 {wall_b/1000:.3f} ms")
print()
print(f"chrome trace 已导出: {TRACE}")
print("用浏览器打开 https://ui.perfetto.dev ，把文件拖进去就能看到时间轴")
