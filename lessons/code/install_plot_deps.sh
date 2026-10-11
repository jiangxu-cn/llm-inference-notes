#!/usr/bin/env bash
# 安装绘制时间轴需要的库
set -uo pipefail
export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"

PY="$HOME/work/venv/bin/python"

uv pip install --python "$PY" matplotlib 2>&1 | tail -4

echo
echo "--- 验证 ---"
"$PY" -c "import matplotlib; print('matplotlib', matplotlib.__version__)"
"$PY" -c "import numpy; print('numpy', numpy.__version__)"
