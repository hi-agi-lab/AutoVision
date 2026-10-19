# 冒烟测试与企业级验收

`scripts/smoke_enterprise.py` 是统一的端到端冒烟入口。它不替代完整测试套件，
而是回答“这套安装在这台机器上，主要路径是否真的能跑通”。

## 运行

```bash
# 快速：训练 -> ONNX -> serving -> registry -> compliance -> 数据快照 -> K8s
python scripts/smoke_enterprise.py

# 完整：额外跑 OpenVINO、TensorRT、DDP、FSDP
python scripts/smoke_enterprise.py --profile full

# 保存 JSON 报告，保留临时产物
python scripts/smoke_enterprise.py --profile full --json smoke-report.json --keep
```

退出码：任一 required 检查 FAIL 返回 1；SKIP 不失败（例如未安装 MLflow）。
`--profile full` 需要 CUDA 才能覆盖 TensorRT/FSDP；缺依赖会显示 SKIP。

## 检查项

| 检查 | 覆盖内容 | 失败含义 |
|---|---|---|
| `environment` | Python/torch/CUDA/GPU/可选后端 | 运行环境不完整 |
| `cli_contract` | `autovision-cli --help` / `list-models` / `model-info` | 安装/CLI 入口问题 |
| `train_detection` | 合成 COCO → 2 step 检测训练 → checkpoint/export/manifest/metrics → `load_task_model.predict`（只验产物，不验 AP） | 训练或产物契约问题 |
| `export_onnx` | ONNX fp32 + ORT 数值对拍 | 导出/对拍问题 |
| `serving` | `/healthz` `/readyz` `/predict` `/metrics`、API key、`/docs` 关闭 | 服务契约/安全问题 |
| `registry` | register → staging → prod（默认门禁）→ current/audit | 注册表/门禁问题 |
| `compliance` | 合规报告 + SBOM + sha256 manifest + 数据集快照 | 合规证据链问题 |
| `data_snapshot` | deep snapshot + verify | 数据版本化问题 |
| `k8s_manifests` | Deployment/Service/HPA/PDB/PVC 解析 | 部署清单问题 |
| `quality` | `ruff check` | 静态质量门禁问题 |
| `openvino` | IR 转换 + 编译后模型 parity | OpenVINO 部署问题 |
| `tensorrt` | fp16 engine build + final-box parity | TensorRT 部署问题 |
| `ddp` | 2-rank torchrun 分类训练一致性 | 分布式训练问题 |
| `fsdp` | 2-rank FSDP：bf16 + 梯度累积 + EMA + checkpoint/export/resume | FSDP 训练问题 |

## 它证明了什么

在本机环境上：

- 训练、导出、推理、服务、注册表、合规报告、数据快照的主要路径可运行；
- ONNX/OpenVINO/TensorRT 的数值对拍通过；
- DDP 与 FSDP 的多进程训练、checkpoint、resume、导出通过；
- API key、默认部署门禁、SBOM、数据漂移检查、K8s 清单形态有效。

## 它不证明什么

- **生产负载/长稳**：没有并发压测、soak、内存泄漏、故障注入；
- **多机 NCCL**：只有 smoke 脚本和 gloo 测试，未在目标集群验证；
- **DeepSpeed/ZeRO/弹性训练**：未实现，`parallel` 会显式拒绝；
- **MLflow 后端**：本机未安装，报告显示 `NOT INSTALLED`；
- **SLA/支持**：支持政策仍是草案，企业 SLA 以合同为准；
- **所有任务/模型**：冒烟只选检测主线；完整矩阵由 `pytest tests` 覆盖。

## 企业级判定建议

冒烟全绿表示“功能验收通过”，不等于“企业级生产就绪”。建议把结论拆成：

1. **功能验收**：`scripts/smoke_enterprise.py --profile full` 全绿；
2. **质量门禁**：`pytest tests`、ruff、bandit（medium/high）、mypy 核心模块、
   coverage 下限通过；
3. **生产验收**：额外完成目标硬件上的多机 NCCL、并发/长稳、故障注入、
   安全评审与法务许可确认。

当前仓库在上述 1/2 项上可用；第 3 项仍依赖部署环境与组织流程。
