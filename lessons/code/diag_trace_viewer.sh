#!/usr/bin/env bash
# 诊断：ui.perfetto.dev 能不能访问 + 本机有哪些可用的时间轴查看器

echo "=== 1. 网络连通性 ==="
for u in https://ui.perfetto.dev https://www.google.com https://pypi.org; do
  code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 12 "$u" 2>/dev/null)
  printf "  %-28s HTTP %s\n" "$u" "${code:-失败}"
done

echo
echo "=== 2. 浏览器 ==="
for p in "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
         "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"; do
  [ -f "$p" ] && echo "  存在: $p" || echo "  不存在: $p"
done

echo
echo "=== 3. 查看 Chrome/Edge 版本（决定 chrome://tracing 还在不在）==="
for p in "/mnt/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" \
         "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe"; do
  if [ -f "$p" ]; then
    printf "  %s : " "$(basename "$p")"
    "$p" --version 2>/dev/null || echo "(取不到)"
  fi
done

echo
echo "=== 4. 本地已有的可视化库 ==="
"$HOME/work/venv/bin/python" - <<'PY'
for m in ("matplotlib", "PIL", "numpy"):
    try:
        mod = __import__(m)
        print(f"  {m:12s} 已安装 {getattr(mod, '__version__', '')}")
    except ImportError:
        print(f"  {m:12s} 未安装")
PY
