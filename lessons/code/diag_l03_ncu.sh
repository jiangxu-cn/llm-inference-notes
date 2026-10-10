#!/usr/bin/env bash
# L03 诊断 2：ncu 在 WSL2 里能不能用？
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"
cd "$HOME/work" || exit 1

echo "############ A. nsys 关于 GPU trace 的原话 ############"
nsys profile --trace=cuda --force-overwrite true -o l03gpu \
  "$HOME/work/venv/bin/python" -c "
import torch
a = torch.randn(512, 512, device='cuda')
b = a @ a
torch.cuda.synchronize()
" 2>&1 | grep -iE 'warn|error|unsupport|not support|skip|fail|unable|gpu trace|kernel' | head -15
echo "(以上为 nsys 输出里提到警告/不支持的行；若为空说明它静默失败)"

echo
echo "############ B. ncu 是否能对 kernel 采样 ############"
ncu --launch-count 1 --launch-skip 3 \
  --kernel-name-base demangled \
  "$HOME/work/venv/bin/python" -c "
import torch
a = torch.randn(1024, 1024, device='cuda')
for _ in range(5):
    b = a @ a
torch.cuda.synchronize()
print('ran')
" 2>&1 | tail -40
