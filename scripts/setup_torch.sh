#!/usr/bin/env bash
# 建立 Python 环境并安装 PyTorch (cu128)。全程不需要 sudo。
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"

echo "=== 1. uv ==="
uv --version || { echo "uv 未安装，中止"; exit 1; }

echo
echo "=== 2. 准备目录 ==="
mkdir -p "$HOME/work"
cd "$HOME/work"
echo "work dir: $HOME/work"

VENV="$HOME/work/venv"

echo
echo "=== 3. 创建 venv (Python 3.12) ==="
rm -rf "$VENV"
uv venv "$VENV" --python 3.12

echo
echo "=== 4a. 安装 torch (cu128) ==="
# 注意：ninja 不在 download.pytorch.org 索引里，必须分开装。
# uv 的解析是原子的：任何一个包找不到，整批都不装（这就是上次失败的原因）
uv pip install --python "$VENV/bin/python" \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch || { echo "!! torch 安装失败"; exit 1; }

echo
echo "=== 4b. 安装 ninja (来自 PyPI) ==="
uv pip install --python "$VENV/bin/python" ninja || { echo "!! ninja 安装失败"; exit 1; }

echo
echo "=== 5. 验证 ==="
"$VENV/bin/python" - <<'PY'
import torch
print("  torch        :", torch.__version__)
print("  cuda version :", torch.version.cuda)
print("  is_available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  device       :", torch.cuda.get_device_name())
    print("  capability   :", torch.cuda.get_device_capability())
    a = torch.randn(2048, 2048, device="cuda")
    b = torch.randn(2048, 2048, device="cuda")
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    print("  matmul 2048^3 OK, sample =", round(float(c[0, 0]), 4))
PY

echo
echo "=== 6. ninja 检查 ==="
"$VENV/bin/ninja" --version || echo "  ninja 未找到"

echo
echo "=== 完成 ==="
