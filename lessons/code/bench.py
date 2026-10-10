#!/usr/bin/env python3
"""L02 · 第一个可信的 benchmark

用法：
    source ~/work/venv/bin/activate      # 激活虚拟环境，之后 python 就是 venv 里的那个
    cd ~/work
    python bench.py

    （不想激活也可以，直接用绝对路径：
        ~/work/venv/bin/python ~/work/bench.py ）

【为什么不能直接敲 `python bench.py`】
    1. Ubuntu 24.04 只有 python3，没有 python。激活 venv 后它会提供 python 这个命令
    2. torch 装在 ~/work/venv 里，系统 python3 里没有 → 直接 import 报 ModuleNotFoundError
    3. 这个文件在 ~/work/ 下，不在 /mnt/c/code/ 下
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

"""
依赖两个库：

  time   Python 标准库。这里只用它一个函数：perf_counter()
  torch  张量计算 + CUDA 接口。本例只用到它的「GPU 张量」和
         「同步 / 打点」API，不涉及任何神经网络。
"""
import time
import torch


"""
本段做两件事：关掉 TF32、准备好被测对象。

torch.backends.cuda.matmul.allow_tf32 = False
    关掉 TF32。TF32 是 Ampere 之后引入的「降精度 FP32」：矩阵乘快约 2 倍，
    但尾数只有 10 位。不显式关掉，你测到的是 TF32 的性能，不是 FP32 ——
    这是"我的数字和别人对不上"最常见的原因。

torch.randn(M, M, device="cuda")
    生成 M×M 的标准正态随机张量。
    device="cuda" 表示直接分配在显存里；不写的话会分配在内存，
    后面的矩阵乘就变成 CPU 运算，完全测不到 GPU。

lambda: <表达式>
    把一个无参表达式包成「无参函数」，方便下面反复调用而不用重复写 a @ b。
    （lambda 里只能放一个表达式，不能放语句）
"""
torch.backends.cuda.matmul.allow_tf32 = False
M = 4096
a = torch.randn(M, M, device="cuda")
b = torch.randn(M, M, device="cuda")
matmul = lambda: a @ b


"""
把一组耗时样本排序，返回 (最小值, 中位数, 平均值, 最大值)。

sorted(ts)
    返回排序后的新 list，不改动原 list。
取中位数用下标 len(ts) // 2（整数除法；样本数为偶数时取中间偏大的那个）。
"""
def stats(ts):
    ts = sorted(ts)
    return ts[0], ts[len(ts) // 2], sum(ts) / len(ts), ts[-1]


"""
反面教材：只统计「提交任务」花掉的时间，不包含 GPU 真正执行的时间。

time.perf_counter()
    返回高精度、单调的时钟读数（秒，float）。
    "单调" = 只会往前走，不受系统时间校准（NTP）影响。
    所以计时必须用它，不能用 time.time()。

为什么这段是错的：
    CUDA 是异步的。fn() 只是把任务提交进队列就立刻返回。
    循环 50 次全部提交完也就几十微秒，所以测到的是 CPU 的提交速度。
    本机实测它比真实值小 200~300 倍。
"""
def bench_naive(fn, rep=50):
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    return (time.perf_counter() - t0) / rep * 1e3


"""
【本项目标准计时方法】perf_counter + 每次 synchronize + 取中位数。

torch.cuda.synchronize()
    阻塞 CPU，直到该 GPU 上所有已提交任务全部执行完毕。
    加在每次 fn() 之后，才能保证这段时间里 kernel 真的跑完了。

warmup 参数
    先空跑 warmup 次但不计时。第一次调用要 170 ms 级（各种初始化），
    是稳态的约 30 倍，不热身会把统计量整个带偏。

返回值里的 1e3
    把秒换算成毫秒。
"""
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


"""
用 GPU 自己的时钟打点。写法更"正统"，但在本机（WSL2）实测偏高约 9.5%，
原因见讲义 L02 附录的实验。所以本项目统一用 bench_sync，这个只作对照。

torch.cuda.Event(enable_timing=True)
    在 GPU 时间轴上打点用的对象。
    enable_timing=True 必须显式写 —— 不写的话 elapsed_time() 永远返回 0。

event.record()
    在当前 stream 上记录一个时间戳。异步提交，不阻塞 CPU。

start.elapsed_time(end)
    返回两个事件之间的毫秒数。必须等两个事件都执行完才能调用。

为什么把 50 对事件一次性建好再循环
    这样循环体里只有 record，不夹带对象创建的开销。
"""
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


"""
主流程：依次跑 4 个实验段并把结果打印出来。涉及上面三个 bench_* 和 stats()。

torch.cuda.get_device_name() / get_device_capability()
    读取显卡型号 / 计算能力版本（本机是 RTX 5070 / (12, 0) = sm_120）。

f"{x:9.4f}"
    f-string 的格式说明符：总宽 9 字符、小数点后 4 位、右对齐。
    对齐是为了让多次运行的数字能竖着比。

torch.empty(n, device="cuda")
    分配 n 个未初始化的 float32，直接放在显存里。
    用 empty 而不是 zeros：反正要整块覆盖写，省掉一次初始化写显存。

tensor.copy_(src)
    原地拷贝：把 src 的内容写进 self。这里借它测显存带宽 ——
    一次 copy 的显存流量 = 读 src + 写 self = 2 × 字节数。
    忘记乘 2 是带宽算错的头号原因。

2 * n * 4 / 1e9
    n 个 float32（4 字节）× 读+写两边 = 总传输字节数，再换算成 GB。

2 * M ** 3
    矩阵乘的浮点运算量：M×M 乘 M×M，每个输出元素做 M 次乘加，
    FLOPs = 2 × M³（乘加各算一次）。
"""

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
