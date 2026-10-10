#!/usr/bin/env python3
"""L01 环境自检：5 项断言。全绿才算 L01 通过。

用法：
    ~/work/venv/bin/python /mnt/c/code/env_check.py
"""
import glob
import os
import shutil
import subprocess
import sys

# ---------- 0. 非交互式 shell 不会加载 ~/.bashrc，手动补 PATH ----------
_cuda_bins = sorted(glob.glob("/usr/local/cuda*/bin"), reverse=True)
if _cuda_bins:
    if _cuda_bins[0] not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _cuda_bins[0] + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("CUDA_HOME", os.path.dirname(_cuda_bins[0]))

# 报错时给的人话提示（子串匹配，全部小写）
HINTS = [
    ("no module named 'torch'", "venv 里没装 torch —— 确认用的是 ~/work/venv/bin/python"),
    ("ninja", "缺 ninja：pip install ninja，或 apt install ninja-build"),
    ("no kernel image", "wheel 的 CUDA < 12.8，或 arch 没对上 → 重装 torch 的 cu128 wheel"),
    ("unsupported gpu architecture", "nvcc 太老（需 >= 12.8），或需 export TORCH_CUDA_ARCH_LIST=12.0"),
    ("command not found", "PATH 里没有该命令 —— 检查 /usr/local/cuda-12.8/bin 在不在 PATH"),
    ("no module named 'torch.utils'", "torch 装的是残缺版本，重装"),
]

results = []


def check(name, fn):
    try:
        results.append((True, name, fn()))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        hint = next((h for k, h in HINTS if k in low), "")
        if hint:
            msg += f"\n      💡 {hint}"
        results.append((False, name, msg))


# ---------- ① 驱动认识显卡 ----------
def c1():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True).stdout.strip()
    name, mem, drv = (x.strip() for x in out.split(","))
    assert "NVIDIA" in name, f"显卡名异常: {name}"
    assert int(drv.split(".")[0]) >= 525, f"驱动 {drv} 过旧（WSL profiler 需 >= 525）"
    return f"{name} | {mem} | 驱动 {drv}"


# ---------- ② nvcc >= 12.8 且认识 sm_120 ----------
def c2():
    nvcc = shutil.which("nvcc")
    assert nvcc, "PATH 里没有 nvcc"
    out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=True).stdout
    ver = [l for l in out.splitlines() if "release" in l][0].split("release")[1].split(",")[0].strip()
    assert tuple(int(x) for x in ver.split(".")[:2]) >= (12, 8), f"nvcc {ver} < 12.8"
    archs = subprocess.run([nvcc, "--list-gpu-arch"], capture_output=True, text=True).stdout.split()
    assert "compute_120" in archs, f"nvcc {ver} 不认识 compute_120（架构表里没有）"
    return f"nvcc {ver} | 支持 compute_120 ✅"


# ---------- ③ PyTorch 是 CUDA 版且 cu >= 12.8 ----------
def c3():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() = False（装成 CPU 版了？）"
    assert torch.version.cuda, "torch.version.cuda 为空 → 装的是 CPU wheel"
    cu = tuple(int(x) for x in torch.version.cuda.split(".")[:2])
    assert cu >= (12, 8), f"torch 的 CUDA {torch.version.cuda} < 12.8"
    return f"torch {torch.__version__} | CUDA {torch.version.cuda}"


# ---------- ④ compute capability 是 Blackwell ----------
def c4():
    import torch
    cap = torch.cuda.get_device_capability()
    assert cap >= (12, 0), f"capability {cap}，期望 >= (12,0)"
    return f"{torch.cuda.get_device_name()} | sm_{cap[0]}{cap[1]}"


# ---------- ⑤ 真跑矩阵乘 + 一个自定义 CUDA kernel ----------
def c5():
    import torch
    from torch.utils.cpp_extension import load_inline

    a, b = (torch.randn(1024, 1024, device="cuda") for _ in range(2))
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()

    src = r"""
#include <torch/extension.h>
__global__ void k(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * 2.0f + 1.0f;      // 边界检查：最重要的 CUDA 语法
}
torch::Tensor f(torch::Tensor x) {
    auto y = torch::empty_like(x);
    int n = x.numel();
    k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    return y;
}
"""
    m = load_inline(
        name="l01_probe",
        cpp_sources="#include <torch/extension.h>\ntorch::Tensor f(torch::Tensor);",
        cuda_sources=src,
        functions=["f"],
        verbose=False,
    )
    x = torch.arange(8, dtype=torch.float32, device="cuda")
    y = m.f(x)
    assert torch.allclose(y, x * 2 + 1), f"自定义 kernel 结果错: {y}"
    return f"matmul 1024^3 OK | 自定义 kernel OK | 首元素 {c[0, 0].item():.2f}"


check("① 驱动认识显卡", c1)
check("② nvcc >= 12.8 且支持 sm_120", c2)
check("③ PyTorch >= cu128", c3)
check("④ compute capability >= (12,0)", c4)
check("⑤ 矩阵乘 + 自定义 kernel", c5)

print("=" * 70)
for ok, name, detail in results:
    print(f"{'✅' if ok else '❌'} {name}: {detail}")
print("=" * 70)

fails = sum(1 for ok, _, _ in results if not ok)
if fails:
    print(f"{fails} 项未通过 —— 把 ❌ 那几行发我")
    sys.exit(1)
print("🎉 L01 通过，环境可以支撑后面 14 周")
