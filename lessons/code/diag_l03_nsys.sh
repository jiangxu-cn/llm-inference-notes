#!/usr/bin/env bash
# L03 诊断：nsys 为什么没抓到 CUDA kernel
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"
cd "$HOME/work" || exit 1

echo "############ 1. 重跑一次，看完整输出里的警告 ############"
nsys profile --trace=cuda,nvtx --force-overwrite true -o l03diag \
  "$HOME/work/venv/bin/python" -c "
import torch
a = torch.randn(1024, 1024, device='cuda')
b = a @ a
torch.cuda.synchronize()
print('ran ok')
" 2>&1 | grep -viE '^\s*$|Collecting|Generating|\[[0-9/]+%|Generated:|^\s+/home' | head -40

echo
echo "############ 2. CUPTI 是否存在 ############"
ls -d /usr/local/cuda-12.8/extras/CUPTI 2>/dev/null && ls /usr/local/cuda-12.8/extras/CUPTI/lib64/ 2>/dev/null | head -8 || echo "!! 没有 CUPTI 目录"

echo
echo "############ 3. nsys 注入库 ############"
ls -l /opt/nvidia/nsight-systems/2024.6.2/target-linux-x64/ 2>/dev/null | grep -iE 'inject|cupti' || echo "（未找到注入库）"

echo
echo "############ 4. 各类报告是否有数据 ############"
for rep in cuda_api_sum cuda_gpu_kern_sum cuda_gpu_mem_time_sum nvtx_sum; do
  n=$(nsys stats --report $rep --format csv l03diag.nsys-rep 2>/dev/null | grep -vc '^$' || true)
  printf "  %-22s 行数=%s\n" "$rep" "$n"
done

echo
echo "############ 5. sqlite 里到底有哪些表 ############"
which sqlite3 >/dev/null 2>&1 && sqlite3 l03diag.sqlite ".tables" 2>/dev/null | tr -s ' ' '\n' | grep -iE 'cuda|nvtx|gpu' | head -20 || echo "（没有 sqlite3 命令）"
