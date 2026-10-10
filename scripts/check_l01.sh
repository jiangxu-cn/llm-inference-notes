#!/usr/bin/env bash
# L01 装备核查（显式设置 PATH，不依赖交互式 shell）

export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$HOME/work/venv/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

echo "===== 1. CUDA 工具链 ====="
printf "  nvcc : "; nvcc --version 2>/dev/null | grep release || echo MISSING
printf "  nsys : "; nsys --version 2>/dev/null | head -1 || echo MISSING
printf "  ncu  : "; ncu --version 2>/dev/null | grep -i version | head -1 || echo MISSING

echo
echo "===== 2. 宿主编译器 ====="
printf "  gcc  : "; gcc --version | head -1
printf "  g++  : "; g++ --version | head -1
printf "  ninja: "; ninja --version

echo
echo "===== 3. nvcc 支持的 arch ====="
nvcc --list-gpu-arch 2>/dev/null | tr '\n' ' '
echo

echo
echo "===== 4. CUDA 安装目录 ====="
ls -d /usr/local/cuda* 2>/dev/null
du -sh /usr/local/cuda-12.8 2>/dev/null | cut -f1 | sed 's/^/  大小: /'

echo
echo "===== 5. venv / torch ====="
VENV="$HOME/work/venv"
if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" -c "
import torch
print('  torch        :', torch.__version__)
print('  cuda version :', torch.version.cuda)
print('  is_available :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  device       :', torch.cuda.get_device_name())
    print('  capability   :', torch.cuda.get_device_capability())
"
else
  echo "  venv 不存在"
fi

echo
echo "===== 6. 磁盘 ====="
df -h / | tail -1
