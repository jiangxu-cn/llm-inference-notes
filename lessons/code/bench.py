#!/usr/bin/env python3
"""L02 · 第一个可信的 benchmark

用法：
    cd ~/work
    ~/work/venv/bin/python bench.py

────────────────────────────────────────────────────────────────────────
【本文件用到的方法 / 函数速查】

time.perf_counter()
    返回高精度、**单调**的时钟读数（秒，float）。
    "单调" = 只会往前走，不受系统时间校准（NTP）影响。
    所以计时必须用它，不能用 time.time()。

torch.backends.cuda.matmul.allow_tf32 = False
    关掉 TF32。TF32 是 Ampere 之后引入的「降精度 FP32」，矩阵乘快约 2 倍，
    但尾数只有 10 位。不显式关掉，你测到的是 TF32 性能，不是 FP32。

torch.cuda.Event(enable_timing=True)
    在 GPU 时间轴上打点的对象。
    enable_timing=True **必须显式写** —— 不写的话 elapsed_time() 永远返回 0。

event.record()
    在当前 stream 上记录一个时间戳（异步提交，不阻塞 CPU）。

start.elapsed_time(end)
    返回两个事件之间的**毫秒数**。必须等两个事件都执行完才能调用。

torch.cuda.synchronize()
    阻塞 CPU，直到该 GPU 上所有已提交任务全部执行完毕。

torch.empty(n, device="cuda")
    分配 n 个未初始化的 float32，直接放在显存里。
    用 empty 而不是 zeros：不需要初始化，省掉一次写显存。

tensor.copy_(src)
    原地拷贝 src 的内容到 self。这里借它测显存带宽（读 src + 写 self）。

lambda: <表达式>
    把一个无参表达式包成「无参函数」，方便反复调用。

sorted(list)
    返回排序后的新 list。取中位数用 ts[len(ts) // 2]。

────────────────────────────────────────────────────────────────────────
【整体流程】

  1. 打印前 6 次调用各自的耗时 —— 演示 warmup 为什么不能省

  2. 同一个 matmul 用三种方法计时并对比：
       bench_naive   不 synchronize       → 测到的是「提交时间」，错的
       bench_sync    perf_counter + sync  → **本项目标准方法**
       bench_event   CUDA Event           → 本机实测偏高约 9.5%

  3. 把时间换算成有物理意义的数字：
       matmul → TFLOPS（算力）
       copy   → GB/s（带宽）

  4. 打印 min / median / mean / max，说明该报哪一个、为什么
────────────────────────────────────────────────────────────────────────
"""

import time

import torch

torch.backends.cuda.matmul.allow_tf32 = False

M = 4096
a = torch.randn(M, M, device="cuda")
b = torch.randn(M, M, device="cuda")
matmul = lambda: a @ b


def stats(ts):
    ts = sorted(ts)
    return ts[0], ts[len(ts) // 2], sum(ts) / len(ts), ts[-1]


def bench_naive(fn, rep=50):
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep * 1e3


def bench_sync(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return stats(ts)


def bench_event(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(rep)]
    for s, e in pairs:
        s.record()
        fn()
        e.record()
    torch.cuda.synchronize()
    return stats([s.elapsed_time(e) for s, e in pairs])


print(f"GPU: {torch.cuda.get_device_name()}  capability={torch.cuda.get_device_capability()}")
print()

print("【1】warmup：前 6 次调用各自的耗时")
for i in range(6):
    t0 = time.perf_counter()
    matmul()
    torch.cuda.synchronize()
    print(f"    iter {i}: {(time.perf_counter() - t0) * 1e3:9.3f} ms")
print()

print("【2】同一个 matmul，三种计时方法（warmup=10, rep=50, 取中位数）")
naive = bench_naive(matmul)
sync_min, sync_med, _, _ = bench_sync(matmul)
ev_min, ev_med, _, _ = bench_event(matmul)
print(f"    ❌ 不 synchronize    : {naive:9.4f} ms   <- 测到的是提交时间")
print(f"    ✅ perf_counter+sync : {sync_med:9.4f} ms   (min {sync_min:.4f})")
print(f"    ⚠️  CUDA Event       : {ev_med:9.4f} ms   (min {ev_min:.4f})")
print(f"    -> naive 把真实时间低估了 {sync_med / naive:.0f} 倍")
print()

print("【3】换算成有物理意义的数字（这才是能写进报告的东西）")
print(f"    matmul {M}^3 FP32 : {2 * M ** 3 / (sync_med / 1e3) / 1e12:6.1f} TFLOPS (理论峰值 ~30)")
n = 64 * 1024 * 1024
x = torch.empty(n, device="cuda")
y = torch.empty(n, device="cuda")
_, copy_med, _, _ = bench_sync(lambda: y.copy_(x))
print(f"    copy 256MB       : {2 * n * 4 / 1e9 / (copy_med / 1e3):6.1f} GB/s   (理论 ~672)")
print()

print("【4】该报哪个统计量（30 次重复）")
ts = []
for _ in range(30):
    t0 = time.perf_counter()
    matmul()
    torch.cuda.synchronize()
    ts.append((time.perf_counter() - t0) * 1e3)
mn, med, mean, mx = stats(ts)
print(f"    min    : {mn:8.4f} ms   <- 最接近「无干扰」的真实性能")
print(f"    median : {med:8.4f} ms   <- 默认用这个，抗离群值")
print(f"    mean   : {mean:8.4f} ms")
print(f"    max    : {mx:8.4f} ms   <- 远大于 median 就说明有干扰")
