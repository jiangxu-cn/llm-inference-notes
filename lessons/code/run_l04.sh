#!/usr/bin/env bash
# 同步代码 + 跑 L04 + 把图拷回 Windows 侧
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"

bash "$HOME/work/sync.sh"
cp -f /mnt/c/code/lessons/code/plot_style.py "$HOME/work/" 2>/dev/null || true

cd "$HOME/work" || exit 1
echo
echo "=== 跑 L04 ==="
"$HOME/work/venv/bin/python" l04_roofline.py

echo
mkdir -p /mnt/c/code/lessons/code/traces
cp -f l04_roofline.png /mnt/c/code/lessons/code/traces/
ls -lh /mnt/c/code/lessons/code/traces/l04_roofline.png
