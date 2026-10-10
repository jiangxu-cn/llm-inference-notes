# LLM 推理 / 高性能优化工程师 · 25 周学习路线

> **起点** 2026-10-12（周一） · **默认投入** 15 h/周（合计约 375 h） · **终点** 2027-04-04
> **终点状态**：3 个能讲 30 分钟的作品 + 能扛住 Infra 岗技术面 + 看得懂 vLLM 生产源码

---

## 0. 这份路线怎么用

只有一条强制规则：

> **每个阶段都有「交付物」和「可量化的验收标准」。交付物不达标，不进入下一阶段。**

- 验收标准全部是**数字**，不是"看完了""理解了""大概懂了"。
- 三件事的顺序不能反：**测量能力 → 算子能力 → 系统能力**。
  先学会"怎么知道它慢在哪"，再学"怎么写快"，最后才是"怎么把一堆算子拼成引擎"。
- 时间不够时，**砍阶段 5（生产框架），绝不砍阶段 2（CUDA 核心）**。阶段 2 是这个岗位的门票。
- 每周日花 20 分钟更新第 12 节的进度表。

---

## 1. 前提假设与目标

### 1.1 能力基线假设

| 能力 | 假设起点 | 本路线中的处理 |
|---|---|---|
| Python | 会用，写过 PyTorch 脚本 | 不专门教，边做边补 |
| C++ | 读得懂，写过简单类 | 阶段 2.5 补底层（内存/多线程/虚函数） |
| Linux | 会用基本命令 | 阶段 0 一次性补齐 |
| CUDA | **零基础** | 阶段 2 从零开始，这是全路线重心 |
| 体系结构 | 模糊 | 阶段 1 建立"瓶颈判断力" |
| 算法/数据结构 | 面试够用或略弱 | 全程每天 1 题保温 |

### 1.2 目标岗位画像（来自 JD）

不是"训练模型的算法工程师"，而是**懂 AI 算法的底层系统工程师**。日常工作流：

```
读论文（FlashAttention / PagedAttention / 量化新方法）
   → 用 CUDA / Triton 把它工程化成算子
   → 做量化 + 图优化塞进推理引擎
   → 部署到 GPU 集群，profiling 压出极致性能
```

三个能力环，缺一个都会在面试里露馅：

| 环 | 能力 | 对应阶段 | 作品集产出 |
|---|---|---|---|
| ① 硬件与算子 | CUDA Kernel、GPU 架构、profiling | 阶段 1–2 | **作品 1：my-kernels** |
| ② 推理机制 | KV Cache、量化、Attention 变体 | 阶段 3 | 作品 2 的前半 |
| ③ 引擎与系统 | 调度、PagedAttention、并行、服务化 | 阶段 4–5 | **作品 2：nano-vLLM 增强**、**作品 3：端到端服务** |

---

## 2. 总览：25 周一张表

| 阶段 | 周次 | 日期 | 主题 | 交付物 | 一句话验收 |
|---|---|---|---|---|---|
| **P0** | W1 | 10/12–10/18 | 环境与测量基线 | `env-baseline.md` | 一条命令复现 baseline 数字 |
| **P1** | W2–W5 | 10/19–11/15 | 体系结构 + GPU 架构 | Roofline 笔记一页 | 口算算术强度并判断 bound 类型 |
| **P2** | W6–W11 | 11/16–12/27 | **CUDA 核心（重心）** | `my-kernels` 仓库 | SGEMM 达 cuBLAS 60%+，FA 省 5× 显存 |
| **P3** | W12–W15 | 12/28–01/24 | LLM 推理机制与量化 | 显存预算计算器 + 量化 kernel | 30 秒报出 max batch 与延迟下限 |
| **P4** | W16–W19 | 01/25–02/21 | nano-vLLM → vLLM | 引擎 fork + 真实功能改进 | 画出一次请求的完整调用链并有数字 |
| **P5** | W20–W22 | 02/22–03/14 | 生产框架与服务化 | 端到端部署 + 压测报告 | 每个性能数字都能追到 profiler 证据 |
| **P6** | W23–W24 | 03/15–03/28 | 冲刺与面试 | 3 个作品的故事化 + 面试题库 | 模拟面试通过 |
| **BUF** | W25 | 03/29–04/04 | 缓冲 / 补债 | — | — |

> ⚠️ **2027 春节为 2 月 6 日**，假期大致落在 W17–W18。这两周按**半量**安排，或直接当缓冲周用。别把最难的阶段排在这两周。

---

## 3. 阶段 0（W1）：环境与测量基线

**为什么它必须排第一**：后面 24 周你说的每一句话都会是"快了 3 倍"，如果没有可靠的测量，那都是幻觉。

### 学习内容
- Linux 速通：`grep/find/awk/sed/htop/nvidia-smi/top -H`、`ssh`、`tmux`、进程与内存查看
- 环境：`uv` 或 `conda` 隔离；PyTorch + CUDA 版本对齐；`nvcc --version` 与 `torch.version.cuda` 一致
- 工具链安装：`nsight-systems`（`nsys`）、`nsight-compute`（`ncu`）、`nvtop`
- 跑通第一次推理：`transformers` 加载 Qwen2.5-0.5B 或 Llama-3.2-1B，贪心解码 128 token

### 动手任务
1. 写一个 `bench.py`：固定 prompt + 固定输出长度 + `torch.cuda.synchronize()` + warmup 10 次 + 计时 50 次取中位数，输出 `prefill 延迟 / decode 每 token 延迟 / 吞吐 tokens/s / 峰值显存`
2. 用 `nsys profile python bench.py` 出第一条 timeline，截图，标出 prefill 和 decode 的位置
3. 记录 GPU 型号、驱动、CUDA、PyTorch、显存、`nvidia-smi -q` 里的显存带宽

### 交付物 `env-baseline.md`
环境清单 + baseline 数字表 + 第一条 timeline 截图。

### 验收标准
- 换一个模型名，改一个参数，能**一条命令**得到同格式的 baseline 数字
- 能解释 timeline 上为什么 decode 阶段有大量小 kernel 和空隙

### 常见坑
- ❌ 直接开始写 CUDA，测出来比 `torch` 慢，但不知道是 kernel 慢还是测量方法错
- ❌ 忘记 warmup，第一次运行包含 CUDA context 初始化和 autotune
- ❌ 用 `time.time()` 测 GPU 而不 `synchronize()`

---

## 4. 阶段 1（W2–W5）：体系结构 + GPU 架构

**目标**：建立"这个操作是计算密集还是访存密集"的判断力。这是所有优化的起点。

### 4.1 存储层次与性能模型（W2–W3）

| 材料 | 读什么 |
|---|---|
| 《深入理解计算机系统》(CSAPP) | **只读 Ch1、Ch5（5.1–5.6 循环优化/CPE）、Ch6（6.2 局部性、6.4 Cache、6.6 存储器层次）** |
| Roofline Model（原论文 + 任意一篇中文解析） | 算术强度 = FLOPs / Bytes；横轴算术强度、纵轴性能；屋脊点位置的含义 |
| 《计算机组成原理》（国家高等教育智慧教育平台） | 只挑 指令流水线、Cache 映射与命中 两节 |

**动手**：写三个 micro-benchmark 画自己的"本地 Roofline 图"
1. 纯计算循环（FMA 密集），测峰值 GFLOPs
2. 大数组顺序遍历，测实际内存带宽 GB/s
3. 矩阵乘不同尺寸，测算术强度与实测性能的关系

### 4.2 GPU 架构（W4–W5）

| 材料 | 重点 |
|---|---|
| NVIDIA A100 / H100 架构白皮书 | SM 结构、warp scheduler、寄存器文件、Shared Memory/L1、L2、HBM、Tensor Core 代际差异 |
| NVIDIA CUDA C Programming Guide 第 3–5 章 | 编程模型 + 硬件实现（warp、占用率） |
| 国产平台概览（昇腾 CANN / 寒武纪） | **只求能对话**：达芬奇架构的 Cube/Vector/Scalar 单元、片上存储手工管理、DMA 搬数 |

**必须自己算出来的数字**（面试会问）：

| 指标 | A100 80GB SXM | H100 SXM |
|---|---|---|
| FP16 Tensor Core（dense） | ~312 TFLOPS | ~989 TFLOPS |
| 显存带宽 | ~2039 GB/s | ~3350 GB/s |
| 屋脊点（算术强度） | ≈ 150 FLOP/Byte | ≈ 295 FLOP/Byte |

> 具体数字以白皮书为准，量级和推导过程才是要记住的。

### 交付物
一页 A4（做成 PDF 或手绘拍照）：
1. A100 vs H100 关键参数对照
2. 常见算子在 Roofline 上的位置：GEMM(prefill)、GEMV(decode)、Softmax、LayerNorm、Elementwise
3. 一句话结论：**为什么 batch=1 的 decode 一定是 memory-bound**

### 验收标准（口头，对着录音说 3 分钟）
- 说出 GPT-2 一层 Attention 在 `batch=1, seq=1`（decode）时的算术强度 ≈ 1 FLOP/Byte，并说明它离屋脊点差多少
- 说出"占用率 100% 不代表快"的一个具体反例

### 常见坑
- ❌ 从 CSAPP 第一页读到最后一页 —— 4 周会变成 3 个月
- ❌ 死记带宽数字，但推不出"decode 延迟下限 = 权重字节数 / 带宽"
- ❌ 只读 CUDA 语法，不理解 warp divergence 和 bank conflict 的硬件原因

---

## 5. 阶段 2（W6–W11）：CUDA 核心 ← **全路线最重要**

**目标**：能独立写出接近硬件上限的 kernel，并知道慢在哪。这 6 周决定你能不能拿到这个岗位。

### 5.1 学习材料（按顺序）

| 顺序 | 材料 | 用法 |
|---|---|---|
| 1 | 《CUDA by Example》 | **快速过**（3 天）。Grid-Block-Warp-Thread 模型、内存模型 |
| 2 | **PMPP《大规模并行处理器程序设计》第 4 版** | **精读 Ch5–Ch12**。这是主教材，含 LLM 章节 |
| 3 | NVIDIA GTC：*How CUDA Programming Works* / *How To Write A CUDA Program* | 跟着第一性原理走一遍，建立直觉 |
| 4 | 《CUDA 并行编程与性能优化》（徐佳宁，清华社） | 与 PMPP 对照做练习，中文表述更顺 |
| 5 | `coder0143/fasterkernels` 仓库 | 参考实现：同一算子的 Triton / TileLang / CUDA 三种写法对比 |

### 5.2 W6：线程模型 + 内存模型 + 与 PyTorch 集成

- [ ] grid/block/warp/thread 的映射；`threadIdx/blockIdx/blockDim/gridDim`
- [ ] warp divergence 是怎么产生的，怎么用 `__shfl_*` 避免
- [ ] Shared Memory、Bank Conflict（32 banks × 4B = 128B 一行）、`__syncthreads()`
- [ ] `cudaStream` / `cudaEvent`，为什么多 stream 能重叠
- [ ] `torch.utils.cpp_extension.load_inline` 把 CUDA 接进 PyTorch（**这一步必须亲手做通**）
- **产物**：一个正确但慢的 `vec_add` + `matrix_transpose`，并写出它的 nsight compute 报告

### 5.3 W7–W9：Kernel 优化三部曲

按这个顺序做，每一步都要有"优化前 vs 优化后"的数字：

**① 归约与 Softmax（W7）**
- naive → 树形归约 → warp shuffle 归约 → 多 block 两段式
- **Online Softmax**（数值稳定的单遍算法）—— 这是 FlashAttention 的地基
- 对比 `torch.softmax`，目标：同尺寸下不慢于 1.5×

**② SGEMM（W8–W9，重头戏）**
按这个阶梯逐级优化，每级都记录 GFLOPs：
```
naive（每个线程一个元素）
 → 全局内存合并访问
 → Shared Memory tiling（分块）
 → Register tiling / thread coarsening（每线程算 8×8）
 → 双缓冲（double buffering）/ cp.async
 →（选做）WMMA / Tensor Core
```
- **目标：FP32 达到 cuBLAS 的 60%–80%**。达不到就继续调，这一步的含金量最高。

**③ 归一化与融合算子（W9 后半）**
- LayerNorm / RMSNorm（一个 block 处理一行，warp reduce）
- Elementwise 融合：`mul + add + GELU` 三算子合一，测量减少的 HBM 读写量
- 对比 PyTorch eager 与 `torch.compile`，看差在哪

### 5.4 W10：性能分析（Profiling）

- [ ] `ncu` 看 Speed of Light：Compute Throughput vs Memory Throughput，一眼定位瓶颈
- [ ] 占用率计算：寄存器数 / Shared Memory / block 数三者如何互相限制
- [ ] `__launch_bounds__`、`-Xptxas -v` 看寄存器压力
- [ ] 用 `ncu --metrics` 定位 bank conflict 与 uncoalesced access
- [ ] `nsys` 看端到端 timeline：kernel 之间的空隙（launch overhead / 同步 / 拷贝）
- **产物**：给 SGEMM 写一份 5 页的性能分析报告（每一级优化 + ncu 截图 + 结论）

### 5.5 W11：FlashAttention 前向

从零实现，分三步走，每步都测：
1. **naive attention**：完整物化 `S = QK^T`（O(N²) 显存）
2. **tiled + online softmax**：分块计算，不物化 S
3. **加 causal mask + 反向（选做）**

- **目标：`seq_len=2048, head_dim=64` 下显存占用比 naive 降低 5× 以上，速度不慢于 naive**
- 对照读 FlashAttention v1 → v2（减少非 matmul FLOPs、更好的并行切分）论文
- 对比 `torch.nn.functional.scaled_dot_product_attention` 的实现

### 交付物（作品集 1）：`my-kernels` 仓库

```
my-kernels/
├── README.md            # 一张总表：每个 kernel 的实测性能 vs 基线
├── 01_reduce/
│   ├── naive.cu  optimized.cu
│   ├── bench.py         # 一键复现
│   └── report.md        # 为什么快了 + ncu 证据
├── 02_sgemm/            # 逐级优化的 5 个版本
├── 03_norm_fusion/
├── 04_flash_attn/
└── tools/bench_utils.py
```

**这个仓库就是你面试时的"作品"，README 的表格是第一印象。**

### 验收标准
- [ ] SGEMM FP32 ≥ cuBLAS 的 60%
- [ ] FlashAttention 前向：seq=2048 显存降低 ≥ 5×
- [ ] 每个 kernel 都有 nsight 截图和一段"瓶颈是什么、我怎么改的、数字变了多少"
- [ ] 能不看笔记讲清楚 shared memory tiling 为什么能提升性能（从 HBM 流量角度）

### 常见坑（这一阶段的坑最致命）
- ❌ **一上来就想读 vLLM 源码**。没有 CUDA 直觉，读源码只会抄，面试一问就露馅
- ❌ 疯狂调参数（block size、tile size）而不理解原因 → 面试无法解释
- ❌ 只写 kernel 不写 benchmark → 无法证明优化有效
- ❌ 用 `printf` 调试 CUDA 而不学 `compute-sanitizer` / `cuda-gdb`

---

## 6. 阶段 3（W12–W15）：LLM 推理机制与量化

**目标**：把 Transformer 的公式和"显存/带宽账本"精确对上。面试里"这个模型能不能跑起来"是必考题。

### 6.1 W12：Transformer 结构细节 + KV Cache

- MHA / MQA / **GQA** 的区别与取舍；RoPE；RMSNorm；SwiGLU
- **KV Cache 记账公式**（必须背下来并能现场推导）：
```
KV bytes = 2 × n_layers × n_kv_heads × head_dim × seq_len × batch × dtype_bytes
权重 bytes = P × dtype_bytes
激活/临时 ≈ 与 batch × seq 相关（prefill 时不可忽略）
```
- **Prefill vs Decode 的本质差异**：prefill 是 compute-bound（大 GEMM），decode 是 memory-bound（GEMV + 读整个 KV Cache）

### 6.2 W13：量化

| 主题 | 必须搞清楚 |
|---|---|
| 基础 | 对称 vs 非对称；per-tensor / per-channel / per-group；scale 与 zero-point |
| 误差 | 量化误差怎么来的，为什么 per-channel 比 per-tensor 好，group size 的权衡 |
| 方法 | GPTQ（二阶 Hessian 误差补偿）、AWQ（激活感知的显著通道保护）、SmoothQuant（把激活的量化难度迁移到权重）、FP8（E4M3/E5M2，per-block scaling） |
| 工程 | W8A16 / W4A16 / W8A8 各自适合什么场景；为什么 KV Cache 量化收益大但风险高 |

**动手**：
- [ ] 手写一个 per-channel 权重量化 + 反量化 kernel（CUDA），对比 `torch` 的浮点结果
- [ ] 用 GPTQ 或 AWQ 量化一个 7B 模型（HuggingFace 生态有现成脚本），测 PPL 变化 + 显存变化 + 速度变化
- [ ] 画一张"位宽 vs PPL vs 显存 vs 速度"的四象限表

### 6.3 W14–W15：论文精读 + 引擎设计思想

| 论文/博客 | 读什么 |
|---|---|
| FlashAttention v1 / v2 / v3 | IO 复杂度分析、recomputation、work partitioning |
| **PagedAttention**（vLLM 论文） | 为什么 KV Cache 碎片率能到 60–80%，block table 如何把浪费降到 <4% |
| **Continuous Batching** | iteration-level 调度 vs request-level 调度的吞吐差异 |
| Speculative Decoding | 原理与加速比上限（选读） |
| [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | **先读这篇博客建立心智模型，再读源码** |

### 交付物
1. `kv_budget.py`：输入模型配置 + 显卡显存，输出权重占用、KV 每 token 占用、max batch size、理论 decode 延迟下限（`权重字节 / 带宽`）
2. 量化 kernel + 对比报告
3. 一篇 1500 字中文笔记：《GQA 到底省了多少 KV Cache，为什么长上下文场景它是刚需》

### 验收标准
- 给你 `Llama-3-8B, 80GB A100, seq=8192`，**30 秒内**报出 max batch size 和 decode 理论延迟下限
- 能解释"为什么 W4A16 比 W8A8 更常用"（权重是访存瓶颈、激活量化误差更敏感）

### 常见坑
- ❌ 只记量化方法名字，说不出"为什么 AWQ 要保护显著通道"
- ❌ 以为 decode 慢是算力不够 —— 其实是带宽不够
- ❌ 跳过显存记账，面试被问"70B 模型单卡能不能跑、要怎么配"直接卡住

---

## 7. 阶段 4（W16–W19）：nano-vLLM → vLLM

**目标**：从 kernel 视角上升到系统视角。这是"算子工程师"和"推理引擎工程师"的分界线。

### 7.1 W16–W17：吃透 nano-vLLM

仓库：[GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)（约 1200 行，实现了 PagedAttention、Continuous Batching、张量并行、CUDA Graph）

四步走，**别跳**：
1. **跑通**：跑起来，和 HF 输出对齐（greedy 逐 token 一致性）
2. **画图**：自己画出架构图 —— Scheduler / BlockManager / Attention Backend / ModelRunner / Sampler 的职责边界和数据流
3. **消融实验**（关键一步）：逐个关掉优化，测吞吐
   | 消融项 | 测什么 |
   |---|---|
   | 关掉 CUDA Graph | 小 batch decode 的吞吐下降多少 → 量化 launch overhead |
   | 关掉 Continuous Batching（改静态 batch） | 混合长度请求下的吞吐差异 |
   | 改 block size（8 / 16 / 32） | 碎片率 vs 调度开销的权衡 |
   | 关掉 PagedAttention（连续 KV） | 并发上限的变化，OOM 出现在哪 |
4. **加一个自己的东西**（三选一）：
   - 把阶段 3 写的量化 kernel 接进去
   - 实现 chunked prefill（可参考 [anjoj0/nano-vllm-v1](https://github.com/anjoj0/nano-vllm-v1)）
   - 实现 prefix caching

### 7.2 W18–W19：读 vLLM 生产源码

**顺序很重要**（顺着数据流读，不要按文件树读）：
```
entrypoints（OpenAI API）
  → LLMEngine / AsyncLLMEngine（请求生命周期）
    → Scheduler（调度策略、chunked prefill）
      → BlockSpaceManager（PagedAttention 的块管理）
        → Worker / ModelRunner（执行）
          → AttentionBackend（FlashAttention / FlashInfer 等后端）
```
- 用 `VLLM_LOGGING_LEVEL=DEBUG` + `nsys` 跑一次，把源码里的阶段名和 timeline 对上
- 遇到不认识的算子，回到阶段 2 的 kernel 知识去看它的实现

### 7.3 张量并行（贯穿 W18–W19）

- 手推 Megatron-LM 的切分方式：MLP 用 column-parallel → row-parallel，Attention 怎么切，为什么最后需要一次 all-reduce
- 用 2 张卡跑一次 `TP=2`，测通信开销占比（`nsys` 看 NCCL kernel）
- 思考题：TP / PP / EP 分别在什么规模下更划算

### 交付物（作品集 2）
- nano-vLLM 的 fork + 一个**真实功能改进**（提 PR 或写好 diff 说明）
- 吞吐对比表：优化前/后 × 不同并发数 × 不同输入输出长度
- 一页架构图（手画也行，但必须自己画）

### 验收标准
- 能画出一次请求从 HTTP 进入到 token 返回的**完整调用链**，每一环都能说出对应的数据结构和耗时量级
- 能解释"为什么 continuous batching 能把吞吐提升数倍"，用你自己的消融数字

### 常见坑
- ❌ fork 了但没真正改任何逻辑（面试会追问"你改的那行为什么必要"）
- ❌ 只读 Scheduler 一个文件，不理解它和 BlockManager 的耦合
- ❌ 用"吞吐提升了"这种模糊表述 —— 必须给并发数和长度分布

---

## 8. 阶段 5（W20–W22）：生产框架与服务化

**目标**：对齐 JD 的"熟悉 TensorRT / TVM / 自研框架"和"框架层研发 + 服务化"。

### 8.1 W20：llama.cpp —— 学 C++ 工程品味

重点观察（而不是读完）：
- **C API 边界**：为什么纯 C 接口、避免虚函数开销
- **ggml 张量库**：内存 arena（`ggml_context`）的设计、图构建与执行分离
- **GGUF 格式与量化**：block-wise 量化的数据结构，亲手跑一次 `quantize` 并对比不同量化档位的 PPL 与速度
- **代码风格**：无异常/RTTI、手工 SIMD、为什么它能跑在从手机到服务器的所有设备上

**产出**：读 `ggml-quants.c` 里一个量化 block 的 dequant 实现，用 CUDA 重写它。

### 8.2 W21：TensorRT-LLM —— 编译期优化

- 架构分层：Python 侧构图 → 编译成 engine → C++ runtime
- **Plugin 机制**：自己写一个简单 plugin，理解它如何嵌入 engine
- **In-flight batching** + **Paged KV Cache** + **CUDA Graph** 如何降低 CPU 开销
- 跑通一个 7B 模型，用 `nsys` 分析 engine 逐层耗时，找出瓶颈层
- 了解 TVM / TileLang / Triton 作为"算子 DSL"的定位差异（写过 Triton 就够对话了）

### 8.3 W22：服务化 + 国产平台

**服务化**：
- `vLLM` 的 OpenAI 兼容 server，或 `Triton Inference Server`（dynamic batching / model ensemble）
- 压测：`locust` 或自写并发脚本，测 **p50 / p95 / p99 延迟** 与 **吞吐-并发曲线**
- K8s + GPU 调度的基本概念：nvidia device plugin、MIG、多实例共享

**国产平台**（只求能对话，不求能写）：
- 昇腾 CANN / AscendCL：达芬奇架构、Cube/Vector/Scalar、片上存储手工管理、DMA
- 思考并写下来：**把一个 CUDA kernel 移植过去，需要改哪些东西？**（编程模型映射、无 warp 概念、显式搬数、算子库对齐）

### 交付物（作品集 3）：端到端部署项目

```
一个 7B 模型的量化部署：
  模型 → 量化 → 引擎 → HTTP 服务 → 压测 → 报告
```
报告必须包含：
1. 环境与配置（GPU 型号、量化方案、并发设置）
2. **延迟-并发曲线**与**吞吐-并发曲线**
3. `nsys` timeline + `ncu` 关键 kernel 的 SOL 分析
4. 你找到的 top-3 瓶颈 + 对应的优化尝试 + 效果（包括失败的尝试）
5. 与 baseline（HF transformers 直接推理）的对比表

### 验收标准
- 报告里**每一个性能数字都能追溯到一次 profiler 证据**
- 能讲清楚"为什么在这个并发点之后吞吐不再增长"（GPU 利用率 / KV Cache 容量 / 调度开销，指哪一个）

### 常见坑
- ❌ 只报一个"吞吐 2000 tokens/s"而没有并发和长度上下文
- ❌ 报告写成实验记录，没有"假设 → 实验 → 结论"的结构

---

## 9. 阶段 6（W23–W24）+ 缓冲（W25）：冲刺与面试

### 9.1 把三个作品故事化

每个作品准备一个 5 分钟版本，结构统一：
```
业务问题 → 我的假设 → 做了什么 → 数字结果 → 为什么有效 → 我踩过的坑
```
准备 3 个"深挖点"（面试官一定会追问细节）：
- my-kernels：SGEMM 从 20% 到 70% cuBLAS，中间哪一级提升最大，为什么
- nano-vLLM：你的改动影响了什么，怎么证明不是噪音
- 部署项目：top 瓶颈是什么，ncu 的哪个指标告诉你的

### 9.2 高频面试题库（按类别自测，答不上来的回去补）

**C++ 底层（15 题）**
- `new/delete` vs `malloc/free`；内存对齐；RAII
- 虚函数表布局、为什么构造函数不能是虚函数、虚析构的必要性
- `std::move` / 右值引用 / 完美转发；`vector` 扩容策略与迭代器失效
- `shared_ptr` 的引用计数与循环引用、`weak_ptr`
- 多线程：`atomic` 内存序、自旋锁 vs 互斥锁、false sharing
- `std::string` SSO、STL 容器底层数据结构与复杂度
- 编译期优化：内联、`constexpr`、LTO

**CUDA（15 题）**
- thread/warp/block/grid 层次；warp divergence 的硬件原因
- Shared Memory 的 bank conflict 是什么，怎么消除（padding 技巧）
- 占用率由什么决定，占用率高一定快吗
- `__syncthreads()` 的语义与误用
- 全局内存合并访问；L1/L2 命中
- `cudaStream` 与异步拷贝、`pinned memory` 为什么快
- CUDA Graph 解决什么问题（量化 launch overhead）
- 如何用 ncu 定位一个 kernel 是 compute-bound 还是 memory-bound

**大模型推理（15 题）**
- Transformer 一层的前向计算量与参数量
- KV Cache 大小公式、GQA 省多少、长上下文的影响
- prefill vs decode 的瓶颈差异
- FlashAttention 为什么省显存、IO 复杂度怎么算
- PagedAttention 解决什么、block table 怎么工作
- Continuous Batching 为什么提升吞吐
- 张量并行的切分方式与通信量
- Speculative Decoding 的加速比上限

**量化（10 题）**
- 对称/非对称、per-tensor/per-channel/per-group 的取舍
- 量化误差来源与评估指标（PPL / 任务准确率）
- GPTQ / AWQ / SmoothQuant 各自解决什么问题
- FP8 的 E4M3 vs E5M2，为什么需要 per-block scaling
- KV Cache 量化的难点

**框架与系统（10 题）**
- vLLM 的整体架构，一次请求的调用链
- TensorRT-LLM 的编译流程与 plugin 机制
- 显存 OOM 的排查思路
- 多卡通信瓶颈怎么定位（NCCL + nsys）
- 精度对齐问题的排查方法论

### 9.3 保温
- 每天 1 道 LeetCode（中等偏上 / 高频），只为不手生
- 每两天投一次简历，把面试当免费的查漏补缺
- 面试后 24 小时内记录被问倒的问题，回去补，写进第 12 节的复盘栏

---

## 10. 每周固定节奏模板（15 h/周）

| 时间 | 时长 | 内容 |
|---|---|---|
| 周一–周五 | 1.5 h × 5 | 学习 + 写代码（选整块时间，不要碎片化） |
| 周六 | 4 h | **动手主战场**：调 kernel / 跑实验 / 写 benchmark |
| 周日 | 4 h | 补债 + 写交付物 + 更新进度表 |

### 每个主题用「三遍法」
1. **跑通**：先把能跑的官方 example 跑起来，建立反馈
2. **费曼**：合上资料，用自己的话写下原理；写不出来就是没懂
3. **测量**：做一次对照实验，拿到自己的数字

> 判断自己有没有学会的标准只有一条：**能不能在没有参考资料的情况下，把它重新写一遍并解释为什么。**

---

## 11. 硬件条件不足怎么办

### 11.1 需要的算力（按阶段）

| 阶段 | 最低要求 | 说明 |
|---|---|---|
| P1–P2 | **任意 NVIDIA GPU**（8GB 够） | kernel 优化不需要大显存 |
| P3 | ≥ 16 GB 显存 | 跑 7B 量化模型 |
| P4 | ≥ 24 GB，最好 2 卡 | 张量并行要 2 卡 |
| P5 | ≥ 24 GB | 端到端部署 |

### 11.2 方案

1. **本地卡（最优）**：RTX 3090/4090 24GB 二手，能覆盖 P1–P5 的 80%
2. **按小时租云 GPU（推荐）**：
   - 国内：AutoDL / 阿里云 / 腾讯云 / 火山引擎，按小时计费，4090/A10 每小时**几元人民币量级**（以平台实时价格为准）
   - 策略：把 P2 的 kernel 开发放本地（或 CPU 侧写好），**只在跑 benchmark 和端到端实验时开机**，按月成本可控
   - 一定要用 `tmux` + 脚本化，避免交互式浪费机时
3. **免费额度**：Colab（T4）、ModelScope 创空间、Kaggle（每周 30h T4/P100）—— 适合做 P1–P2 的基础练习
4. **国产卡租用**：昇腾 910 / 寒武纪 MLU 有云上按需实例，**既能练手又能写进简历的"国产平台适配经验"**

### 11.3 完全没 GPU 的降级路线

仍然可以做，只是进度变慢、部分作品受限：
- P1 全部可做（Roofline 推导、白皮书、架构分析）
- P2 改用 **Triton**（CPU 上可读可写，用 Colab 验证），但**手写 CUDA 部分必须找到卡补上**，这是门票
- P3 的显存记账、量化原理、论文精读全部可做
- P4 读源码可做，跑实验要租卡
- P5 用 ModelScope / 云上按小时做

---

## 12. 三条时间变体

| 变体 | 周期 | 投入 | 调整 |
|---|---|---|---|
| **冲刺** | 12 周 | 25 h/周 | P1 压到 2 周；P2 压到 4 周（只做 SGEMM + FlashAttention，跳过归约/融合）；P5 只做部署不做国产平台 |
| **标准**（本路线） | 25 周 | 15 h/周 | 按本文档执行 |
| **稳扎** | 36 周 | 8–10 h/周 | P2 拉长到 10 周，每个 kernel 都做到极致；P4 读完整 vLLM 源码 |

**如果只能保留一件事**：把 SGEMM 从 naive 优化到 cuBLAS 的 70%。这一件事能证明你具备这个岗位的全部核心素质。

---

## 13. JD 技能覆盖映射表

| JD 要求 | 覆盖阶段 | 覆盖程度 |
|---|---|---|
| C++（3 年 Linux 经验） | P0 / P2 / P5（llama.cpp 阅读） | ⚠️ 需额外补：C++ 底层八股（第 9.2 节） |
| Python | 全程 | ✅ |
| 算法与数据结构 | 全程保温 | ✅ |
| Linux 环境 | P0 | ✅ |
| CUDA 编程 / 手写 Kernel | **P2** | ✅ 核心 |
| GPU 架构与工作原理 | P1 / P2 | ✅ |
| 国产 GPU 平台 | P5.3 | ⚠️ 只能到"能对话"级别 |
| LLM 算子原理（Attention/GEMM/LayerNorm） | P2 / P3 | ✅ |
| 算子融合 | P2.3 | ✅ |
| 模型量化（INT8/FP16/INT4, GPTQ/AWQ/SmoothQuant） | P3.2 | ✅ |
| 模型压缩与蒸馏 | P3.2（蒸馏仅提及） | ⚠️ 需补：知识蒸馏基础 |
| 分布式集群推理（TP/PP） | P4.3 | ✅ TP 深入，PP 了解 |
| 推理框架（TensorRT/TVM/自研） | P4 / P5 | ✅ |
| 性能分析工具链（nsys/ncu/perf） | P0 / P2.4 / P5 | ✅ 这是本路线的强项 |
| 大规模框架研发 / 模型服务化 / K8s + GPU | P5.3 | ⚠️ 应用级，非源码级 |
| 协同沟通（精度 vs 性能平衡） | P3.2 / P5 报告 | ✅ 用"精度-速度权衡表"作为证据 |

> ⚠️ 标记项是这份路线的**已知缺口**，面试前需要单独补：C++ 底层八股、知识蒸馏、K8s 实操。

---

## 14. 资料优先级总表

### 必读（做透）
| 资料 | 用途 |
|---|---|
| **PMPP《大规模并行处理器程序设计》第 4 版** | CUDA 主教材，含 LLM 章节 |
| NVIDIA CUDA C Programming Guide | 权威参考 |
| NVIDIA A100 / H100 架构白皮书 | GPU 架构 |
| CSAPP Ch5–Ch6 | 存储层次与性能优化 |
| [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) | 推理引擎全景 |
| [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm) | 引擎心智模型 |
| FlashAttention v1/v2 论文 | Attention 优化核心 |
| PagedAttention（vLLM）论文 | 显存管理核心 |

### 选读（按兴趣深入）
| 资料 | 用途 |
|---|---|
| 《CUDA by Example》 | 快速入门 |
| 《CUDA 并行编程与性能优化》（徐佳宁，清华社） | 中文对照练习 |
| NVIDIA GTC：*How CUDA Programming Works* | 建立直觉 |
| [coder0143/fasterkernels](https://github.com/coder0143/fasterkernels) | Triton/TileLang/CUDA 三写法对比 |
| [zimei11/ai-infra-notes](https://github.com/zimei11/ai-infra-notes) | 量化/稀疏化/内存管理/调度的中文串联 |
| llama.cpp 的 `ggml-quants.c` | C++ 工程品味 |
| TensorRT-LLM 官方文档 | 编译期优化 |
| 《计算机组成原理》（国家高等教育智慧教育平台） | 指令流水线与 Cache |

### 查阅（用到再翻）
Cornell CS6120 / 各类 LLM 系统课程、vLLM 官方文档、国内各大厂 Infra 团队技术博客。

---

## 15. 进度追踪表

每周日更新。`✅` 完成 / `🟡` 进行中 / `❌` 未开始。

| 周 | 日期 | 主题 | 状态 | 本周实际投入 | 复盘备注 |
|---|---|---|---|---|---|
| W1 | 10/12 | 环境与基线 | ❌ | | |
| W2 | 10/19 | 存储层次 + Roofline | ❌ | | |
| W3 | 10/26 | CSAPP Ch5–6 + micro-bench | ❌ | | |
| W4 | 11/02 | GPU 架构白皮书 | ❌ | | |
| W5 | 11/09 | 架构笔记 + 验收口述 | ❌ | | |
| W6 | 11/16 | CUDA 线程/内存模型 + PyTorch 集成 | ❌ | | |
| W7 | 11/23 | 归约 + Online Softmax | ❌ | | |
| W8 | 11/30 | SGEMM 逐级优化（上） | ❌ | | |
| W9 | 12/07 | SGEMM（下）+ LayerNorm 融合 | ❌ | | |
| W10 | 12/14 | nsight 性能分析 | ❌ | | |
| W11 | 12/21 | FlashAttention 前向 | ❌ | | |
| W12 | 12/28 | Transformer 细节 + KV 记账 | ❌ | | |
| W13 | 01/04 | 量化原理 + kernel | ❌ | | |
| W14 | 01/11 | FlashAttention / PagedAttention 论文 | ❌ | | |
| W15 | 01/18 | Continuous Batching + 交付物 | ❌ | | |
| W16 | 01/25 | nano-vLLM 跑通 + 画架构图 | ❌ | | |
| W17 | 02/01 | nano-vLLM 消融（春节半量） | ❌ | | |
| W18 | 02/08 | 加自己的功能（春节半量） | ❌ | | |
| W19 | 02/15 | vLLM 源码 + 张量并行 | ❌ | | |
| W20 | 02/22 | llama.cpp / ggml / GGUF | ❌ | | |
| W21 | 03/01 | TensorRT-LLM | ❌ | | |
| W22 | 03/08 | 服务化 + 压测 + 国产平台 | ❌ | | |
| W23 | 03/15 | 作品故事化 + 面试题自测 | ❌ | | |
| W24 | 03/22 | 模拟面试 + 复盘 | ❌ | | |
| W25 | 03/29 | 缓冲 / 补债 | ❌ | | |

### 里程碑检查点

| 检查点 | 时间 | 判定 | 不达标怎么办 |
|---|---|---|---|
| M1 | W5 末 | 能口算算术强度、判断 bound 类型 | 停一周，重做 micro-benchmark |
| M2 | W11 末 | SGEMM ≥ 60% cuBLAS | **必须解决**，可以延后一周，但不能跳过 |
| M3 | W15 末 | 30 秒报出 max batch + 延迟下限 | 补显存记账练习 |
| M4 | W19 末 | 画出完整调用链并有数字 | 补消融实验 |
| M5 | W22 末 | 端到端部署报告完成 | 砍掉国产平台部分保交付 |

---

## 16. 风险管理：最常见的 6 种失败模式

| 失败模式 | 症状 | 对策 |
|---|---|---|
| **过早读源码** | 读 vLLM 三周，能复述架构，但写不出一个 softmax kernel | 严格执行 P2 先于 P4 |
| **只看不写** | 视频看了 50 小时，仓库还是空的 | 每个主题必须产出可运行代码 |
| **没有测量** | 所有结论都是"感觉快了" | P0 的 bench 工具贯穿全程 |
| **陷入参数调优** | 花两周调 tile size，说不出原理 | 每次优化先写下假设，再测，再解释 |
| **贪多求全** | 同时学 TVM + Triton + CUDA + 昇腾 | 一次只深入一个，其余到"能对话"即可 |
| **后期才准备面试** | 作品很好但讲不出来 | W23 就开始把作品故事化，边做边记要点 |

---

## 17. 如果只能记住三句话

1. **测量优先**：没有 profiler 证据的优化都是猜。
2. **P2 是门票**：手写 CUDA 到接近 cuBLAS，比读过十个框架源码更有说服力。
3. **一切都记账**：显存账、带宽账、FLOPs 账。能算账的人，面试官会当成自己人。
