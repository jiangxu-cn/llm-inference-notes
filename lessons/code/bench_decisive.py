#!/usr/bin/env python3
"""L02 决定性实验：sync 与 event 差的 7%，到底来自「计时器」还是「有没有间隙」？

设计：
  A. sync         : perf_counter + 每次 synchronize   （有间隙）
  B. event + sync : Event 计时 + 每次 synchronize     （有间隙）
  C. event + burst: Event 计时 + 30 次背靠背          （满载，无间隙）

如果 B ≈ A 且 B < C  -> 差异来自「有没有间隙」（满载降频），不是计时器
如果 B ≈ C 且 A < B  -> 差异来自计时器本身
"""
import statistics
import time
import torch

torch.backends.cuda.matmul.allow_tf32 = False
a = torch.randn(4096, 4096, device="cuda")
b = torch.randn(4096, 4096, device="cuda")
WARMUP, REP = 10, 30


def matmul():
    return a @ b


def method_a(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REP):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def method_b(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(REP):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return ts


def method_c(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(REP)]
    for i in range(REP):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(st, en)]


print("热身中 ...")
for _ in range(300):
    matmul()
torch.cuda.synchronize()
print()

print(f"{'method':<22} {'median':>9} {'min':>9} {'max':>9}")
print("-" * 52)
rows = {}
for name, fn in [("A. sync + perf_counter", method_a),
                 ("B. Event + 每次 sync", method_b),
                 ("C. Event + 背靠背满载", method_c)]:
    ts = fn(matmul)
    med, mn, mx = statistics.median(ts), min(ts), max(ts)
    rows[name[0]] = med
    print(f"{name:<22} {med:>9.4f} {mn:>9.4f} {mx:>9.4f}")

print()
print("解读：")
print(f"  A vs B 差 {abs(rows['A'] - rows['B']) / rows['A'] * 100:5.1f}%  -> 计时器本身的差异")
print(f"  B vs C 差 {abs(rows['C'] - rows['B']) / rows['B'] * 100:5.1f}%  -> 有没有间隙的差异")
print()
print("如果 A≈B 且 B<C：两种方法测的是「隔离执行的峰值性能」vs「满载的稳定性能」，")
print("两个数字都对，但回答的是不同问题 —— 所以报告里必须写清用的是哪种。")
