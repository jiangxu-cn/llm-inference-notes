#!/usr/bin/env python3
"""L02 · 第一个可信的 benchmark

用法：~/work/venv/bin/python bench.py
"""
import time
import torch

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False   # 保证测的是真 FP32，不是 TF32

print(f"GPU: {torch.cuda.get_device_name()}  capability={torch.cuda.get_device_capability()}")
print()

# ============ 1. 为什么开头几次调用特别慢 ============
print("【1】开头几次调用为什么特别慢（warmup 的意义）")
a = torch.randn(4096, 4096, device=DEV)
b = torch.randn(4096, 4096, device=DEV)


def matmul():
    return a @ b


for i in range(6):
    t0 = time.perf_counter()
    matmul()
    torch.cuda.synchronize()
    print(f"    iter {i}: {(time.perf_counter() - t0) * 1e3:9.3f} ms")
print()

# ============ 2. 三种计时方法对比 ============


def bench_naive(fn, rep=50):
    """❌ 错误示范：不 synchronize"""
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep * 1e3


def bench_sync(fn, warmup=10, rep=50):
    """✅ perf_counter + synchronize，取中位数"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def bench_event(fn, warmup=10, rep=50):
    """✅✅ CUDA Event（推荐做法）"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
    for i in range(rep):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ts[len(ts) // 2]


n_naive = bench_naive(matmul)
n_sync = bench_sync(matmul)
n_event = bench_event(matmul)

print("【2】同一个 matmul，三种计时方法")
print(f"    ❌ 不 synchronize    : {n_naive:9.4f} ms   <- 测到的是提交时间")
print(f"    ✅ perf_counter+sync : {n_sync:9.4f} ms")
print(f"    ✅✅ CUDA Event       : {n_event:9.4f} ms")
print(f"    -> naive 把真实时间低估了 {n_sync / n_naive:.0f} 倍")
print()

# ============ 3. 换算成有物理意义的数字 ============
flops = 2 * 4096 ** 3
print("【3】换算成物理量（这才是能写进报告的数字）")
print(f"    matmul 4096^3 : {flops / (n_event / 1e3) / 1e12:6.1f} TFLOPS FP32 (理论峰值 ~30)")

n = 64 * 1024 * 1024                      # 64M 个 float = 256 MB
x = torch.empty(n, device=DEV)
y = torch.empty(n, device=DEV)
ms_copy = bench_event(lambda: y.copy_(x))
gb = 2 * n * 4 / 1e9                      # 读 256MB + 写 256MB
print(f"    copy 256MB    : {gb / (ms_copy / 1e3):6.1f} GB/s      (理论 ~672)")
print()

# ============ 4. 中位数 vs 平均值 vs 最小值 ============
ts = []
for _ in range(30):
    t0 = time.perf_counter()
    matmul()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
ts.sort()
mean = sum(ts) / len(ts)
print("【4】取哪个统计量？（30 次重复）")
print(f"    min    : {ts[0]:8.4f} ms   <- 最接近「无干扰」的真实性能")
print(f"    median : {ts[len(ts) // 2]:8.4f} ms   <- 默认用这个，抗离群值")
print(f"    mean   : {mean:8.4f} ms   <- 被个别慢的拖高")
print(f"    max    : {ts[-1]:8.4f} ms   <- 看它就知道有没有被干扰")
