#!/usr/bin/env python3
"""L03 · 用 nsys 看见 kernel 之间的空隙

四个阶段，每个阶段用 NVTX 打上标签，方便在 timeline 上定位：
  A. 背靠背提交 20 个 kernel，最后只 sync 一次   -> GPU 应该一直是满的
  B. 每提交 1 个 kernel 就 sync 一次              -> GPU 中间应该有空隙
  C. 从可分页内存拷到显存（5 次）
  D. 从锁页内存拷到显存（5 次）                  -> 应该明显更快

用法：
    nsys profile --trace=cuda,nvtx --force-overwrite true -o l03 python nsys_demo.py
"""
import torch

N = 20
a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")


def kernel():
    return a @ b


# ---------- warmup：不做的话第一段会被 context 初始化污染 ----------
for _ in range(5):
    kernel()
torch.cuda.synchronize()

# ---------- A. 背靠背提交，只在最后 sync 一次 ----------
torch.cuda.nvtx.range_push("A_burst_20kernels")
for _ in range(N):
    kernel()
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()

# ---------- B. 每次提交后立刻 sync ----------
torch.cuda.nvtx.range_push("B_sync_each")
for _ in range(N):
    kernel()
    torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# ---------- C. 可分页内存 -> 显存 ----------
n = 32 * 1024 * 1024                      # 32M 个 float = 128 MB
pageable = torch.empty(n, dtype=torch.float32)
torch.cuda.nvtx.range_push("C_pageable_H2D")
for _ in range(5):
    d = pageable.to("cuda")
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

# ---------- D. 锁页内存 -> 显存 ----------
pinned = torch.empty(n, dtype=torch.float32).pin_memory()
torch.cuda.nvtx.range_push("D_pinned_H2D")
for _ in range(5):
    d = pinned.to("cuda", non_blocking=True)
torch.cuda.synchronize()
torch.cuda.nvtx.range_pop()

print("done")
