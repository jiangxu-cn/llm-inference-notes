#!/usr/bin/env bash
# 对比 /mnt/c（9p）与 ~/work（ext4）的 I/O 与编译开销
set -uo pipefail

WSL_DIR="$HOME/work/_iotest"
WIN_DIR="/mnt/c/Users/jiang/AppData/Local/Temp/_iotest"

echo "############ 0. 文件系统类型 ############"
df -T "$HOME/work" /mnt/c 2>/dev/null | awk 'NR==1 || /work|\/mnt\/c/'
echo
echo "内核挂载方式:"
mount | grep -E 'on / type|on /mnt/c type' | head -3

echo
echo "############ 1. Python I/O 对比 ############"
echo "[WSL 虚拟盘 ext4]"
"$HOME/work/venv/bin/python" /mnt/c/code/lessons/code/io_bench.py "$WSL_DIR"
echo
echo "[Windows 盘 9p]"
"$HOME/work/venv/bin/python" /mnt/c/code/lessons/code/io_bench.py "$WIN_DIR"

echo
echo "############ 2. nvcc 编译对比（输出到不同目录）############"
cat > /tmp/_k.cu <<'CUEOF'
#include <cstdio>
__global__ void k(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 2.0f + 1.0f;
}
int main() {
    float* d;
    cudaMalloc(&d, 1024 * sizeof(float));
    k<<<4, 256>>>(d, 1024);
    cudaDeviceSynchronize();
    cudaFree(d);
    printf("ok\n");
    return 0;
}
CUEOF

for D in "$WSL_DIR" "$WIN_DIR"; do
  mkdir -p "$D"
  t0=$(date +%s%N)
  nvcc -arch=sm_120 --keep --keep-dir "$D" -o "$D/k.out" /tmp/_k.cu 2>/dev/null
  t1=$(date +%s%N)
  printf "  输出到 %-46s %6d ms\n" "$D" $(( (t1 - t0) / 1000000 ))
  rm -rf "$D"
done

rm -rf "$WSL_DIR" "$WIN_DIR" /tmp/_k.cu /tmp/_k.* 2>/dev/null
echo
echo "（测试目录已清理）"
