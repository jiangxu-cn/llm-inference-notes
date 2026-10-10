#!/usr/bin/env python3
"""L03 诊断 3：torch.profiler 在 WSL2 里能不能拿到 CUDA kernel 时间？

如果它能拿到，就说明 CUPTI 的 activity 通道是通的，只是 nsys 的用法有问题；
如果它也拿不到，说明 WSL2 的 GPU 活动追踪整体不可用。
"""
import torch
from torch.profiler import ProfilerActivity, profile

a = torch.randn(2048, 2048, device="cuda")
b = torch.randn(2048, 2048, device="cuda")

for _ in range(3):
    c = a @ b
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()

print("=== key_averages（按 cuda_time_total 排序）===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12))

print()
print("=== 每个 CUDA kernel 的原始事件 ===")
evts = [e for e in prof.events() if e.device_type.name == "CUDA"]
print(f"CUDA 事件总数: {len(evts)}")
for e in evts[:8]:
    print(f"  {e.name[:48]:<50} dur={e.duration_time_ns/1e3:8.3f} us")
