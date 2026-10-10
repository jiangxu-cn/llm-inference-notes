#!/usr/bin/env python3
"""测量：用 torch 编译一个自定义 CUDA kernel，构建产物放不同盘上要多久

这是 W4–W6 每天都要做的事（load_inline / cpp_extension），
也是 /mnt/c 慢不慢最直接的判据。

用法：
    python ext_bench.py <构建产物目录> <扩展名>
"""
import os
import shutil
import sys
import time

build_dir, ext_name = sys.argv[1], sys.argv[2]

shutil.rmtree(build_dir, ignore_errors=True)
os.makedirs(build_dir)
os.environ["TORCH_EXTENSIONS_DIR"] = build_dir

from torch.utils.cpp_extension import load_inline

"""
TORCH_EXTENSIONS_DIR
    告诉 torch 把编译产物放哪。默认在 ~/.cache/torch_extensions（ext4）。
    这里改成目标目录，用来对比两个盘。

load_inline(name, cpp_sources, cuda_sources, functions)
    现场编译一个 C++/CUDA 扩展并加载回来。
    name 必须唯一 —— 同名会直接复用缓存，测不到编译时间。
    cpp_sources 放声明，cuda_sources 放定义和 __global__ 函数。
    functions 指定要暴露给 Python 的函数名。
"""
CUDA_SRC = r"""
#include <torch/extension.h>
__global__ void k(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 2.0f + 1.0f;
}
torch::Tensor f(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""

t0 = time.perf_counter()
load_inline(
    name=ext_name,
    cpp_sources="#include <torch/extension.h>\ntorch::Tensor f(torch::Tensor);",
    cuda_sources=CUDA_SRC,
    functions=["f"],
    verbose=False,
)
ms = (time.perf_counter() - t0) * 1e3

"""
统计构建过程产生了多少文件、总共多少字节 ——
用来解释"为什么慢"：文件越多，9p 的元数据往返代价越大。
"""
n_files = n_bytes = 0
for root, _, files in os.walk(build_dir):
    for fn in files:
        n_files += 1
        n_bytes += os.path.getsize(os.path.join(root, fn))

print(f"    编译耗时 {ms:8.0f} ms   |  产生 {n_files:4d} 个文件 / {n_bytes / 1e6:6.1f} MB")
