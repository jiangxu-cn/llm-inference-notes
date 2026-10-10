#!/usr/bin/env bash
# 对比：torch 编译自定义 CUDA kernel，构建产物放 ext4 vs 9p
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"

PY="$HOME/work/venv/bin/python"
SCRIPT=/mnt/c/code/lessons/code/ext_bench.py
WSL_DIR="$HOME/work/_exttest"
WIN_DIR="/mnt/c/Users/jiang/AppData/Local/Temp/_exttest"

echo "########## ext4（WSL 虚拟盘）##########"
"$PY" "$SCRIPT" "$WSL_DIR" ext_wsl

echo
echo "########## 9p（Windows 盘）##########"
"$PY" "$SCRIPT" "$WIN_DIR" ext_win

rm -rf "$WSL_DIR" "$WIN_DIR"
echo
echo "（测试目录已清理）"
