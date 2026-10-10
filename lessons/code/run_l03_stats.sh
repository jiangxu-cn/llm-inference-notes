#!/usr/bin/env bash
# L03 · 从 nsys-rep 里提取报告
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"
cd "$HOME/work" || exit 1

echo "############ 1. GPU kernel 汇总 ############"
nsys stats --report cuda_gpu_kern_sum --format table l03.nsys-rep 2>/dev/null | head -20

echo
echo "############ 2. 显存拷贝汇总 ############"
nsys stats --report cuda_gpu_mem_time_sum --format table l03.nsys-rep 2>/dev/null | head -18

echo
echo "############ 3. NVTX 阶段汇总 ############"
nsys stats --report nvtx_sum --format table l03.nsys-rep 2>/dev/null | head -20

echo
echo "############ 4. kernel trace 的 CSV 表头 ############"
nsys stats --report cuda_gpu_trace --format csv l03.nsys-rep 2>/dev/null | head -3
