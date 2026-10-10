#!/usr/bin/env python3
"""L02 终审：用一个「不可能错」的方法当裁判，看 A 和 B 谁是真的。

裁判方法：连续跑 1000 个 matmul 再统一 synchronize。
  总墙钟时间 / 1000 = 单个 kernel 的真实平均耗时（CPU launch 14.8us << 5.7ms，被完全掩盖）
"""
import statistics
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")


def mm():
    return a @ b


# ---- 裁判 ----
for _ in range(20):
    mm()
torch.cuda.synchronize()

N = 1000
t0 = time.perf_counter()
for _ in range(N):
    mm()
torch.cuda.synchronize()
t1 = time.perf_counter()
judge = (t1 - t0) / N * 1e3
print(f"裁判（{N} 个连续 kernel 的平均值）: {judge:.4f} ms\n")


# ---- 两种被检验的方法（同样 1000 次，只换计时方式）----
def method_a(n):
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        mm()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def method_b(n):
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        mm()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts), min(ts)


n = 100
a_med, a_min = method_a(n)
b_med, b_min = method_b(n)

print(f"{'方法':<18} {'median':>9} {'min':>9} {'vs 裁判':>10}")
print("-" * 50)
print(f"{'裁判(连续满载)':<18} {judge:>9.4f} {judge:>9.4f} {'—':>10}")
print(f"{'A. sync+perf':<18} {a_med:>9.4f} {a_min:>9.4f} {(a_med - judge) / judge * 100:>+9.1f}%")
print(f"{'B. Event+sync':<18} {b_med:>9.4f} {b_min:>9.4f} {(b_med - judge) / judge * 100:>+9.1f}%")
print()
print("离裁判更近的那个，才是可信的绝对数字。")
