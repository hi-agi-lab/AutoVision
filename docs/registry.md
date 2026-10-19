# 私有模型注册表与部署门禁

`autovision.registry.ModelRegistry` 是文件系统后端：不可变版本 +
dev/staging/prod + `current` 指针 + 审计日志 + 部署门禁。

## CLI

```bash
export AUTOVISION_REGISTRY_ROOT=/srv/autovision-registry

autovision-cli registry action=register file=best.pt name=det v=1.0.0
autovision-cli registry action=list
autovision-cli registry action=promote name=det v=1.0.0 to=staging gates=all
autovision-cli registry action=promote name=det v=1.0.0 to=prod gates=all
autovision-cli registry action=current name=det stage=prod
autovision-cli registry action=rollback name=det stage=prod to=0.9.0 reason="regression"
autovision-cli registry action=audit
```

- `gates=all` 运行标准门禁；`gates=none` 仅做 sha 校验与晋级（测试/紧急通道）。
  Python API 的 `promote(gates=None)` 默认运行标准门禁（fail-closed）；
  只有显式传 `gates=[]` 才跳过。
- `actor=` 写入审计日志；默认读 `AUTOVISION_ACTOR`，否则 `local`。
- 读写路径（register / promote / get / list / rollback / current / audit）使用
  跨进程文件锁；读操作不会看到半拷贝条目。注册 key 拒绝 `.`、`..`、路径分隔符、
  NUL 和空白键。
- MLflow 后端：注册/晋级会随产物保存并取回 model card / postprocess sidecar，
  `promote(gates=...)` 对取回的副本执行同一套门禁（默认门禁同样 fail-closed），
  `get` 重新校验 sha256，`list_entries` 跳过同 registry 中的非 AutoVision 模型版本。

## 门禁

| gate | 检查 |
|---|---|
| `Sha256Gate` | 晋级副本哈希与注册 manifest 一致 |
| `ModelCardGate` | 模型卡存在、`validate()` 通过、文件 sha 与卡一致 |
| `LicenseGate` | 权重许可在部署 allowlist（默认宽松许可集） |
| `BenchmarkGate` | 模型卡有正的 `coco_map` 与至少一条延迟记录 |
| `ParityGate` | `<file>.parity.json`（或 `parity_report` 指向的报告）的 `passed` 为 true；metadata `parity_report_sha256` 可选绑定 |
| `DataSnapshotGate` | metadata `data_snapshot` 指向存在的数据快照 |

注册时模型卡（`<file>.modelcard.json`）、后处理规格
（`<file>.postprocess.json`）与对拍报告（`<file>.parity.json`）会随产物复制并记录
sha256，门禁读取注册表内的副本。`parity_passed=true` 只在显式使用
`ParityGate(require_report=False)` 时兼容，默认不再接受自报标志。

## Python

```python
from pathlib import Path

from autovision.registry import ModelRegistry, default_gates

# 对拍报告必须随产物提供（注册时会复制并记录 sha256）。
Path("best.pt.parity.json").write_text('{"passed": true}')

registry = ModelRegistry("/srv/autovision-registry")
entry = registry.register("best.pt", name="det", version="1.0.0")
registry.promote(name="det", version="1.0.0", to_stage="staging",
                 gates=default_gates())
current = registry.current(name="det", stage="staging")
print(current.path, current.sha256)
```

## 回滚与审计

- `rollback` 只移动 `current` 指针，不复制/删除不可变版本。
- `audit.jsonl` 追加 register/promote/rollback 记录（时间、actor、版本、
  stage、原因）；不修改历史。
- `current.json` 原子写入（临时文件 + rename）。

## 边界

- 文件系统后端没有多租户/远程鉴权；私有部署请放在受控目录/网络。
- 门禁读取模型卡与 metadata；缺失证据即失败，不猜。
- 签名/HSM/OCI 分发不在当前实现内，可按企业需求叠加。
