#!/usr/bin/env bash
# 把仓库里的代码同步到 ~/work
#
# 为什么要有这一步：
#   WSL 访问 /mnt/c（Windows 盘）走 9p 协议，I/O 慢好几倍。
#   所以「仓库」放 /mnt/c/code（方便 git 同步），「运行副本」放 ~/work。
#   每次改了仓库里的代码，跑一次这个脚本即可。
#
# 用法：
#   bash ~/work/sync.sh

set -e

SRC=/mnt/c/code/lessons/code
DST="$HOME/work"

mkdir -p "$DST"
cp -f "$SRC"/*.py "$DST"/
cp -f /mnt/c/code/env_check.py "$DST"/
cp -f "$SRC"/traces/*.json "$DST"/ 2>/dev/null || true

echo "已同步到 $DST："
ls -1 "$DST"/*.py
