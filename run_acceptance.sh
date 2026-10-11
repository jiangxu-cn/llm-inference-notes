#!/usr/bin/env bash
# 恢复后的最终验收

export PATH="/usr/local/cuda-12.8/bin:$HOME/.local/bin:$PATH"

echo "===== 1. 默认用户 ====="
echo "  whoami: $(whoami)   HOME: $HOME"

echo
echo "===== 2. /mnt/c 挂载是否正常 ====="
if [ -d /mnt/c/code ]; then
  echo "  /mnt/c/code 可访问，文件数: $(ls -1 /mnt/c/code | wc -l)"
else
  echo "  !! /mnt/c/code 不可访问"
fi

echo
echo "===== 3. L01 五项验收（env_check.py）====="
"$HOME/work/venv/bin/python" /mnt/c/code/env_check.py
RC=$?
echo "  退出码: $RC"

echo
echo "===== 4. git 仓库在 WSL 里能否访问 ====="
if [ -d /mnt/c/code/.git ]; then
  echo "  /mnt/c/code/.git 存在"
  head -1 /mnt/c/code/.git/refs/heads/main
fi
