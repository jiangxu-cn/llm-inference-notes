#!/usr/bin/env bash
# 重新采集 trace 并渲染成 PNG
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"

PY="$HOME/work/venv/bin/python"
cd "$HOME/work" || exit 1

sed -i 's/\r$//' /mnt/c/code/lessons/code/l03_render.py
cp -f /mnt/c/code/lessons/code/l03_render.py "$HOME/work/"

echo "=== 1. 重新采集 trace ==="
"$PY" l03_timeline.py 2>&1 | grep -viE 'UserWarning|_warn_once'

echo
echo "=== 2. 渲染成 PNG ==="
"$PY" l03_render.py

echo
echo "=== 3. 拷回 Windows 侧 ==="
mkdir -p /mnt/c/code/lessons/code/traces
cp -f l03_timeline.png /mnt/c/code/lessons/code/traces/
cp -f l03_trace.json  /mnt/c/code/lessons/code/traces/
ls -lh /mnt/c/code/lessons/code/traces/
