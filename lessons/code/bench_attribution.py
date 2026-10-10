#!/usr/bin/env python3
"""L02 追因实验：A 与 B 的差值是「固定开销」还是「成比例」？

做法：把每次测量的工作量从 1 个 matmul 加到 20 个。
  - 若 (B-A) 基本恒定        -> 固定开销（每次测量的启动/事件成本）
  - 若 (B-A)/A 基本恒定      -> 速率差异（时钟/功耗状态）
"""
import statistics
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")


def mm():
    return a @ b


def method_a(fn, warmup=5, rep=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def method_b(fn, warmup=5, rep=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


# 纯 CPU 侧的 launch 成本
for _ in range(20):
    mm()
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 200
for _ in range(N):
    mm()
t1 = time.perf_counter()
torch.cuda.synchronize()
print(f"纯 CPU launch 成本: {(t1 - t0) / N * 1e6:.1f} us/次\n")

print(f"{'工作量':>6} {'A(sync)':>10} {'B(Event)':>10} {'B-A':>9} {'(B-A)/A':>9}")
print("-" * 50)
prev = None
for reps in [1, 2, 5, 10, 20]:
    def work(r=reps):
        for _ in range(r):
            mm()
    A = method_a(work)
    B = method_b(work)
    print(f"{reps:>4} xmm {A:>10.3f} {B:>10.3f} {B - A:>9.3f} {(B - A) / A * 100:>8.1f}%")
