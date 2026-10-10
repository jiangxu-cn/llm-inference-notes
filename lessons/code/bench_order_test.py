#!/usr/bin/env python3
"""L02 附加实验：两种计时方法差 8%，到底是方法差异还是时钟漂移？

做法：交替跑 A/B/A/B，看差值是否稳定。
"""
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")


def matmul():
    return a @ b


def bench_sync(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2], min(ts), max(ts)


def bench_event(fn, warmup=10, rep=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(st, en))
    return ts[len(ts) // 2], ts[0], ts[-1]


print("先热身 30 秒等效负载，让时钟稳定 ...")
for _ in range(300):
    matmul()
torch.cuda.synchronize()
print()

print(f"{'round':>5} | {'sync median':>12} {'sync min':>9} | {'event median':>13} {'event min':>10} | {'diff%':>7}")
print("-" * 74)
for r in range(4):
    s_med, s_min, _ = bench_sync(matmul)
    e_med, e_min, _ = bench_event(matmul)
    diff = (e_med - s_med) / s_med * 100
    print(f"{r:>5} | {s_med:>12.4f} {s_min:>9.4f} | {e_med:>13.4f} {e_min:>10.4f} | {diff:>+6.1f}%")

print()
print("结论看两列 min：如果两种方法的 min 很接近，说明差异只来自统计口径/离群值；")
print("如果 min 也差很多，说明两种方法测的确实不是同一段时间。")
