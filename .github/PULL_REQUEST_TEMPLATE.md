# Pull Request

## 动机

<!-- 这个 PR 解决什么问题；关键决策与取舍；关联的 issue / 设计记录。 -->

## 变更摘要

<!-- 一段话说清改了什么。面向用户的可见变更请勾选下方 CHANGELOG 项。 -->

## 验证

提交前逐项确认（与 CI 同一套门禁，逐条命令见 `CONTRIBUTING.md`）：

- [ ] `ruff check src tests scripts`
- [ ] `pytest tests -q`
- [ ] `python scripts/check_provenance.py` 与 `python scripts/check_layering.py`
- [ ] 改动了文档：`python scripts/check_docs_links.py` 与 `mkdocs build --strict`
- [ ] 改动了权重注册表：`scripts/gen_model_zoo.py --check`
- [ ] 有意变更了配方 / 行为：金标准快照（`tests/golden/`）已按流程重录并说明原因
- [ ] 面向用户的可见变更：`CHANGELOG.md` 的 `[Unreleased]` 段已记录

## 数值 / 行为证据

<!-- 行为或数值变更必须给出证据：对拍读数（bit-equal 优先 / 容差）、
     基准读数（附复现脚本）或短程趋势数据。仅“测试通过”不足以支撑数值类改动。 -->

## 许可与溯源

- [ ] 新增代码文件带 Apache-2.0 版权头；无禁用标记（溯源门禁自证）
- [ ] 新依赖已过许可审查（AGPL 类仅 extras 隔离，不进核心依赖、不模块级 import）
- [ ] 新增权重资产已钉 revision + sha256 + 许可；受限许可默认不下载
- [ ] 涉及第三方代码 / 权重：`NOTICE` 已更新，重打包附逐位忠实性对拍
