# 并行训练与规模边界

本页记录训练侧的并行能力、已实现路径与仍属平台工程的边界。状态以代码与
CI 门禁为准，不以路线图承诺为准。

## 已实现

| 能力 | 入口 | 验证 |
|---|---|---|
| DDP（单机多卡/多机，torchrun） | 默认；`torchrun --nproc_per_node=N` | 2-rank CPU+gloo 端到端（`AUTOVISION_DDP_TESTS=1`）；NCCL/多机需目标硬件 |
| FSDP（任务训练） | `trainer_args={"parallel": "fsdp"}` | 2-rank FSDP（单卡共享 GPU）端到端：训练、验证、full-state checkpoint、导出、resume |
| 梯度检查点 | `trainer_args={"gradient_checkpointing": true}` | `tests/test_gradient_checkpointing.py`（自定义 ViT/GTR blocks + HF 原生 API） |
| AMP（fp16/bf16） | `precision=16-mixed/bf16-mixed` | CPU 单测 + CUDA 门控用例 |
| 断点续训 | `checkpoint=` / `resume_interrupted=True` | DDP 与 FSDP full-state 用例 |
| 梯度累积 | `gradient_accumulation_steps` | 入口校验 + 训练用例 |
| HPO（随机/网格/ASHA） | `autovision-cli hpo` | 真实训练 + 复用指纹回归 |

### FSDP 行为与约束

```bash
torchrun --standalone --nproc_per_node=8 -m autovision.cli ...
# trainer_args={"parallel": "fsdp", "val_every_num_steps": 100}
```

- **仅任务训练**：`TaskEngine.fit_task` 支持 FSDP；pretrain 仍走 DDP。
- **CUDA + 多进程**：FSDP 要求 `WORLD_SIZE >= 2` 且 CUDA；单进程/CPU 会在入口
  报可读错误。
- **整模型 shard**：当前 wrap 整个 `TrainModel`，没有按 transformer block
  auto-wrap；超大模型后续应接 `auto_wrap_policy` 以进一步降低单层物化峰值。
- **Checkpoint/resume**：`last`/`best` 使用 `FullStateDictConfig` +
  `FullOptimStateDictConfig` 在 rank0 汇总完整模型/优化器状态；resume 时所有
  rank 加载同一 full state。`save_last`/`save_best` 语义与 DDP 相同。
- **导出**：rank0 从 full state 生成普通 `exported_models/*.pt`，可被
  `load_task_model` 直接加载（测试已覆盖）。
- **验证**：validation 直接调用 task model，FSDP 下用
  `summon_full_params(writeback=False)` 临时还原完整参数。
- **EMA**：`ModelEMA` 保留完整参数副本；FSDP 下更新 EMA 需要一次 full-param
  gather。默认每步更新（`AUTOVISION_FSDP_EMA_INTERVAL=1`，正确但较慢）；
  大模型可调大间隔换取吞吐，EMA 新鲜度相应下降。
- **rank0 写失败**：checkpoint 写入后所有 rank 走一个 `all_reduce` 失败标志，
  rank0 磁盘/权限错误会中止整个 job，而不是让其他 rank 在下一次集合通信上
  挂死；进程组默认 1800s 超时（`AUTOVISION_DDP_TIMEOUT_SEC`）。

## 未实现（明确边界）

- **DeepSpeed / ZeRO / 弹性训练**：未接入引擎。`parallel` 只接受 `ddp` /
  `fsdp`；其他值会显式报错。需要 ZeRO 分片优化器状态、DeepSpeed 的
  checkpoint/通信 scheduler 与弹性 rendezvous 时，请在外层使用对应框架。
- **多机 NCCL / 故障注入**：仓库提供 `scripts/nccl_smoke.py` 和 2-rank gloo
  测试；NCCL、网络分区、节点故障恢复仍需在目标集群演练。
- **张量并行 / 序列并行**：未实现。

## 多机部署检查清单

0. 先跑 `torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=<node0>:29500 scripts/nccl_smoke.py`，确认 NCCL 初始化与 all-reduce 正常。
1. 每个节点 `torchrun --nnodes=... --nproc_per_node=... --rdzv_backend=c10d` 启动；`devices=auto` 读取 `LOCAL_RANK`。
2. 共享输出目录只允许 rank0 写（引擎已按 `is_global_rank_zero` 门禁），并确认
   rank0 写失败会通过 checkpoint barrier 中止所有 rank。
3. 用 `AUTOVISION_DETERMINISTIC=1` 做一次可复现性验收（会变慢）。
4. 大模型先开 `gradient_checkpointing`，再评估 FSDP；需要更细粒度分片时在外层
   接 FSDP auto-wrap。
