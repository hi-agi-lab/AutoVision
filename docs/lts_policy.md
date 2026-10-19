# AutoVision LTS 政策

> 本文是支持承诺的政策草案；具体合同条款以企业协议为准。

## 版本节奏

- 每月一个 MINOR/PATCH 发布（必要时加安全 PATCH）。
- 每 **6 个月** 选一个 MINOR 作为 **LTS**（例如 `0.3 LTS`、`0.6 LTS`）。
- LTS 从发布日起：
  - **12 个月** 安全修复；
  - **前 6 个月** 完整缺陷修复；
  - 后 6 个月仅安全/严重回归修复。

## LTS 分支规则

- 分支：`release/<major>.<minor>`（从对应 tag 拉出）。
- LTS 只接受：
  - 安全修复；
  - 数据损坏 / 崩溃 / 精度回归；
  - 目标硬件/后端的关键兼容修复。
- 新功能只进主线 MINOR；不 backport 功能。
- 每个 backport 需要：复现、修复、回归测试、CHANGELOG `Security`/`Fixed` 记录。

## 弃用与移除

- 公共 API（`autovision.*` 顶层导出、CLI 命令/参数、配置字段）先在 MINOR
  标注 + 运行时警告，至少跨一个 MINOR 才可移除。
- LTS 分支不引入弃用；已有弃用只按原计划移除，不移除 LTS 稳定面。

## EOL

- 提前 **3 个月** 公告 EOL；EOL 后分支只读，不再发补丁。
- 安全 EOL 版本可提供付费延长支持（企业协议）。

## 安全

- 安全问题通过私下渠道报告（见支持政策）；确认后按 CVSS 评估补丁窗口。
- 严重问题：目标 7 天内发 PATCH；一般问题随月度 PATCH。
- PyPI 版本不可覆盖；严重缺陷使用 **Yank** + 新 PATCH。

## 支持矩阵

每个 LTS 页面登记：Python 版本、PyTorch/CUDA 版本、TensorRT/OpenVINO
版本、通过门禁的后端与硬件。升级依赖前先跑 LTS CI 矩阵。
