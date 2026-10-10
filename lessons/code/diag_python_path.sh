#!/usr/bin/env bash
# 诊断：为什么 `python bench.py` 失败

echo "=== 1. 系统里有哪些 python ==="
ls -l /usr/bin/python* 2>/dev/null || echo "  (无)"

echo
echo "=== 2. 系统 python3 里有 torch 吗 ==="
python3 -c "import torch" 2>&1 | tail -1

echo
echo "=== 3. venv 里有 torch 吗 ==="
"$HOME/work/venv/bin/python" -c "import torch; print('  torch', torch.__version__)"

echo
echo "=== 4. /mnt/c/code 下有 bench.py 吗 ==="
ls /mnt/c/code/*.py 2>/dev/null || echo "  (没有，只有 lessons/code/ 和 scripts/ 里才有)"

echo
echo "=== 5. ~/work 下的脚本 ==="
ls -1 "$HOME/work"/*.py 2>/dev/null || echo "  (无)"

echo
echo "=== 6. 激活 venv 之后 python 是谁 ==="
source "$HOME/work/venv/bin/activate"
which python
python -c "import torch; print('  激活后 python 能看到 torch', torch.__version__)"
