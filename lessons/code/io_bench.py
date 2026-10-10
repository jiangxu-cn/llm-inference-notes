#!/usr/bin/env python3
"""测量 /mnt/c（Windows 盘，走 9p）与 ~/work（WSL 虚拟盘，ext4）的 I/O 差异

用法：
    python io_bench.py <要测试的目录>
"""
import os
import shutil
import sys
import time

d = sys.argv[1]

"""
先清空目标目录，避免上一次的残留文件影响结果（尤其是"建文件"这一项，
如果目录里已有同名文件，覆盖和新建的开销不一样）。
"""
shutil.rmtree(d, ignore_errors=True)
os.makedirs(d)

"""
统一的计时包装：跑一个函数，打印它花了多少毫秒。

time.perf_counter()
    高精度单调时钟（秒）。测 I/O 用它，不用 time.time()。

(fn() 之后) * 1e3
    秒 -> 毫秒
"""
def timed(fn, label):
    t0 = time.perf_counter()
    fn()
    ms = (time.perf_counter() - t0) * 1e3
    print(f"    {label:22s} {ms:9.1f} ms")


"""
写一个 512 MB 的大文件。

open(path, "wb")
    以二进制写模式打开，'w' 会截断已有内容。
f.write(chunk) 写 1 MB 的零字节块，循环 512 次。
f.flush() + os.fsync(f.fileno())
    flush 只把数据从 Python 缓冲推到操作系统；
    fsync 才强制刷到磁盘。不 fsync 的话测到的是"写进页缓存"的速度，不是真 I/O。
"""
def write_big():
    with open(f"{d}/big.bin", "wb") as f:
        chunk = b"\0" * (1 << 20)
        for _ in range(512):
            f.write(chunk)
        f.flush()
        os.fsync(f.fileno())


"""
建 500 个 4 KB 的小文件 —— 模拟编译器产生大量中间文件（.o / .ptx / .cubin）的场景。

这一项最能量出 9p 的真实代价：每建一个文件都是一次跨系统的元数据往返，
而传输的数据量（总共才 2 MB）微不足道。
"""
def many_small():
    for i in range(500):
        with open(f"{d}/f{i}", "w") as f:
            f.write("x" * 4096)


"""
顺序读回那个 512 MB 文件。

while f.read(1 << 20)
    每次读 1 MB，读到空 bytes 就结束（空 bytes 是 falsy）。
"""
def read_big():
    with open(f"{d}/big.bin", "rb") as f:
        while f.read(1 << 20):
            pass


"""
删掉整个目录 —— 模拟清理构建产物。这一项也全靠元数据操作。
"""
def clean():
    shutil.rmtree(d)


print(f"测试目录: {d}")
timed(write_big, "顺序写 512 MB")
timed(many_small, "建 500 个 4 KB 文件")
timed(read_big, "顺序读 512 MB")
timed(clean, "删除整个目录")
