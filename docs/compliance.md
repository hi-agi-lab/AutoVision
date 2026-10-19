# 企业合规报告

`autovision-cli compliance-report` 把企业评审需要的证据汇编成
`compliance.json` + `compliance.md`：

- 框架版本 / commit / Python / 平台；
- 已安装依赖的版本与许可（未知许可显式 `UNKNOWN`，不静默放过）；
- 模型文件 sha256、模型卡（许可 / 指标 / 延迟 / 训练口径）；
- 数据快照（`data-snapshot` 产物）；
- 问题清单（缺失模型卡、sha 不一致、许可待复核等）。

## 用法

```bash
autovision-cli compliance-report \
    model=runs/det/exported_models/exported_best.pt \
    snapshot=qa/data.snap.json out=compliance/
```

- `include_dependencies=false` 跳过依赖扫描（快速检查模型/数据证据）。
- `no_timestamp=true` 生成可复现报告（不写 `generated_at`）。
- `strict=true` 在存在 error 时退出码 1（适合 CI/发布门禁）。
- `fail_on=error|warning|none` 指定门禁级别（优先级高于 `strict`）；
  `fail_on=warning` 把未知/待复核许可、未验证数据快照等 warning 也作为失败。
- 报告同时写出 `compliance.sbom.json`（CycloneDX 1.5）与
  `compliance.manifest.json`（三个报告文件的 sha256），供审计复核。

## 判定

- `error`：模型无许可、模型卡 sha 与文件不一致、模型卡非法、数据快照与内容
  不一致等；
- `warning`：缺模型卡、依赖许可未知/待复核、无数据快照、快照无法重算等；
- `info`：未提供快照等提示。

数据快照会用 `verify_dataset_snapshot` 重新计算；数据移动导致无法重算记
warning，内容漂移记 error。报告是**证据汇编**，不是法律意见；许可最终结论
由法务确认。默认允许清单为常见宽松许可（Apache-2.0 / MIT / BSD / ISC /
MPL-2.0）；企业可在代码中传入自定义 allowlist。

## 边界

- 不读取受限权重内容，只读文件哈希与旁车元数据；
- 不联网；
- 依赖许可来自已安装包元数据，缺失/自定义条款会显示 `UNKNOWN` 或原文。
