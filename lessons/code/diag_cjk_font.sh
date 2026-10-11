#!/usr/bin/env bash
# 找一个能显示中文的字体给 matplotlib 用

echo "=== 1. WSL 系统里有没有中文字体 ==="
if command -v fc-list >/dev/null 2>&1; then
  fc-list :lang=zh 2>/dev/null | head -5 || echo "  (fc-list 查不到)"
else
  echo "  fc-list 命令不存在"
fi
ls /usr/share/fonts/ 2>/dev/null

echo
echo "=== 2. Windows 自带的中文字体（WSL 可以直接读）==="
ls -lh /mnt/c/Windows/Fonts/ 2>/dev/null | grep -iE 'msyh|simhei|simsun|deng|msjh|mingliu' | head -10

echo
echo "=== 3. 逐个测试 matplotlib 能不能加载 ==="
"$HOME/work/venv/bin/python" - <<'PY'
from matplotlib import font_manager as fm

cands = [
    "/mnt/c/Windows/Fonts/msyh.ttc",      # 微软雅黑
    "/mnt/c/Windows/Fonts/msyhl.ttc",     # 微软雅黑 Light
    "/mnt/c/Windows/Fonts/simhei.ttf",    # 黑体
    "/mnt/c/Windows/Fonts/simsun.ttc",    # 宋体
    "/mnt/c/Windows/Fonts/Deng.ttf",      # 等线
]
for p in cands:
    try:
        fm.fontManager.addfont(p)
        print(f"  OK   {p}  -> {fm.FontProperties(fname=p).get_name()}")
    except Exception as e:
        print(f"  FAIL {p}  -> {type(e).__name__}: {e}")
PY
