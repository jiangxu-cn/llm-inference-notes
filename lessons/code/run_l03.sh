#!/usr/bin/env bash
# L03 · 跑 nsys profile
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"

cd "$HOME/work" || exit 1

echo "=== 同步脚本 ==="
sed -i 's/\r$//' /mnt/c/code/lessons/code/nsys_demo.py
cp -f /mnt/c/code/lessons/code/nsys_demo.py "$HOME/work/nsys_demo.py"
ls -l "$HOME/work/nsys_demo.py"

echo
echo "=== nsys 版本 ==="
nsys --version || { echo "nsys 不可用"; exit 1; }

echo
echo "=== 开始 profile ==="
nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite true \
  -o l03 \
  "$HOME/work/venv/bin/python" "$HOME/work/nsys_demo.py" 2>&1 | tail -25

echo
echo "=== 产物 ==="
ls -lh l03.nsys-rep 2>/dev/null || echo "!! 没有生成 .nsys-rep"
