#!/usr/bin/env bash
# 跑 L01 验收

export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$HOME/work/venv/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"

VENV="$HOME/work/venv"

echo "===== 补装 numpy（torch 会警告缺它）====="
uv pip install --python "$VENV/bin/python" numpy 2>&1 | tail -3

echo
echo "===== 运行 env_check.py ====="
"$VENV/bin/python" /mnt/c/code/env_check.py
RC=$?

echo
echo "退出码: $RC"
