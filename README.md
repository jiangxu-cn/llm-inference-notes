# LLM 推理 / 高性能优化 · 学习档案

零基础 → CUDA 算子工程师的 **15 周**学习路线与实操记录。
目标岗位：大模型推理 / 底层性能优化工程师。

**仓库**：https://github.com/jiangxu-cn/llm-inference-notes

> **起点** 2026-10-12 · **投入** 26 h/周 · **终点** 2027-01-24
> **终点状态**：3 个能讲 30 分钟的作品（`my-kernels` / nano-vLLM 增强 / 端到端部署）+ 能扛住 Infra 岗技术面

---

## 硬件与环境

| 项目 | 实测 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070（12 GB，**sm_120**，capability (12, 0)） |
| 驱动 | 617.42（CUDA UMD 13.4） |
| 环境 | WSL2 + Ubuntu 24.04.2 LTS（内核 6.6.87.1） |
| CUDA Toolkit | 12.8（V12.8.93，`--list-gpu-arch` 含 `compute_120`） |
| 宿主编译器 | gcc / g++ 13.3.0 |
| 框架 | torch 2.11.0+cu128 / numpy 2.5.3 |
| Profiler | nsys 2024.6.2.225 / ncu 2025.1.1.0 |
| 虚拟环境 | `~/work/venv`（Python 3.12.3） |

环境验收脚本：`env_check.py`（5 项断言，含自定义 CUDA kernel 编译并运行）

---

## 目录

| 文件 | 说明 |
|---|---|
| [`llm-inference-roadmap.md`](llm-inference-roadmap.md) | 25 周完整版学习路线（阶段拆解 / 验收标准 / 面试题库 / JD 覆盖映射） |
| [`user-level.md`](user-level.md) | **个人水平档案**（水平快照 / 15 周个性化路线 / 基础漏洞处置表 / 教学协议 / 进度日志） |
| [`skill-checklist.md`](skill-checklist.md) | 技能自评清单（60 项，0–4 分锚点） |
| [`lessons/`](lessons/) | 逐讲讲义 |
| [`scripts/`](scripts/) | 环境搭建与验收脚本 |

### 讲义

| 讲 | 主题 | 核心收获 |
|---|---|---|
| [L01](lessons/L01-环境自检.md) | 打通 RTX 5070 环境 | sm_120 的三层结构（驱动 / 工具链 / 框架）、第一个 CUDA kernel |
| [L02](lessons/L02-可信的benchmark.md) | 第一个可信的 benchmark | 用「连续 1000 kernel 当裁判」推翻教科书结论 |

---

## ⭐ 本项目标准测量协议（L02 确立）

用「连续 1000 个 kernel」当独立裁判做校验后发现：**本机 CUDA Event 逐次计时系统性偏高 9.5%**，
而 `perf_counter` + `synchronize` 只偏高 1.5%（`min` 几乎完全吻合裁判值 5.6886 ms）。

```python
def bench(fn, warmup=10, rep=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median": ts[len(ts) // 2], "min": ts[0], "max": ts[-1]}
```

**三条纪律**
1. 同一份报告只用一种计时方法
2. 数字必须带方法：`6.09 ms (perf_counter+sync, warmup=10, rep=50, median)`
3. 换到原生 Linux 后第一件事：重跑裁判实验

**基线（RTX 5070）**

| 项目 | 实测 | 理论 | 达成率 |
|---|---|---|---|
| matmul 4096³ FP32 | 21.0 TFLOPS | ~30 TFLOPS | 70% |
| 256 MB copy | 577.8 GB/s | ~672 GB/s | 86% |
| 首次调用开销 | 204 ms vs 稳态 6.16 ms | — | 33× |

---

## 用法

```bash
# 环境自检
~/work/venv/bin/python env_check.py

# 跑某讲的代码
~/work/venv/bin/python lessons/code/bench.py
```

---

## 日常同步

凭据走 **GitHub CLI**（`gh auth setup-git`），不需要手动管理 token。

```bash
# 每讲结束后
git add -A
git commit -m "L03: nsys 第一条 timeline"
git push
```

> 从 WSL 里操作 `C:\code` 也可以，路径是 `/mnt/c/code`。
> 但注意：**代码和编译产物请放 Linux 侧的 `~/work`** —— WSL 访问 `/mnt/c` 走 9p 协议，I/O 慢好几倍。

---

## 进度

| 周 | 讲 | 状态 |
|---|---|---|
| W1 | L01 环境自检 | ✅ |
| W1 | L02 第一个可信的 benchmark | ✅ |
| W1 | L03 nsys 第一条 timeline | ⏳ |
| W2–W15 | 见 [user-level.md](user-level.md) 第 7 节 | ⏳ |
