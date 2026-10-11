#!/usr/bin/env bash
# 验证恢复回来的 Ubuntu 环境是否完好

echo "===== 1. 系统 ====="
grep PRETTY_NAME /etc/os-release
uname -r
echo "user: $(whoami)   home: $HOME"

echo
echo "===== 2. GPU 透传 ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -2

echo
echo "===== 3. CUDA 工具链 ====="
export PATH="/usr/local/cuda-12.8/bin:$PATH"
printf "  nvcc : "; nvcc --version 2>/dev/null | grep release || echo "MISSING"
printf "  nsys : "; nsys --version 2>/dev/null | head -1 || echo "MISSING"
printf "  ncu  : "; ncu --version 2>/dev/null | grep -i version | head -1 || echo "MISSING"

echo
echo "===== 4. 宿主编译器 ====="
printf "  gcc  : "; gcc --version 2>/dev/null | head -1 || echo "MISSING"
printf "  g++  : "; g++ --version 2>/dev/null | head -1 || echo "MISSING"
printf "  ninja: "; ninja --version 2>/dev/null || echo "MISSING"

echo
echo "===== 5. Python / torch ====="
VENV="$HOME/work/venv"
if [ -x "$VENV/bin/python" ]; then
  "$VENV/bin/python" - <<'PY'
import torch
print("  torch        :", torch.__version__)
print("  cuda version :", torch.version.cuda)
print("  is_available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device       :", torch.cuda.get_device_name())
    print("  capability   :", torch.cuda.get_device_capability())
try:
    import matplotlib, numpy
    print("  matplotlib   :", matplotlib.__version__)
    print("  numpy        :", numpy.__version__)
except Exception as e:
    print("  缺库:", e)
PY
else
  echo "  !! venv 不存在: $VENV"
fi

echo
echo "===== 6. ~/work 里的脚本 ====="
ls -1 "$HOME/work"/*.py 2>/dev/null || echo "  (没有 .py)"
echo "  --- 其他 ---"
ls -1 "$HOME/work" 2>/dev/null | grep -v '\.py$' | head -10

echo
echo "===== 7. 磁盘 ====="
df -h / | tail -1
