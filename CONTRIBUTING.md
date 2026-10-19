# 贡献指南

感谢参与 AutoVision。本框架的维护标准较高：**每个变更都要能被机器验证**——
门禁脚本、对拍测试与协议化基准是评审的硬通货。请先读完本页再动手。

适用范围：bug 修复、功能实现、文档、数据集 / 权重接入、测试与门禁增强。
范围边界：不接收任何受限许可权重的下载捷径、不接收未注明出处的第三方代码
（见「许可与溯源红线」）。

## 开发环境

Python ≥ 3.10，以 pip 安装为准（与 CI 同款）：

```bash
pip install -e ".[dev,export,serving,video,transformers]"   # 测试 + 导出 + 服务/视频依赖（CI 同款）
pip install -e ".[docs]"         # mkdocs 站点构建（改文档时需要）
pytest tests -q                  # 全量测试（CPU 可跑）
```

`uv.lock` 由公共 PyPI 生成，可用 `uv lock --check` 验证复现性；CI 与发布门禁
以 pip 路径为准。重新生成时必须显式使用公共索引：
`UV_DEFAULT_INDEX=https://pypi.org/simple uv lock`。

## 提交前必须通过的门禁

与 CI 同一套门禁（命令与 `.github/workflows/ci.yml` 一致；`gen_model_zoo
--check` 由测试套件在 CI 内执行）：

```bash
ruff check src tests scripts             # lint（规则集见 pyproject.toml）
bandit -r src/autovision -ll -ii -q      # SAST（medium/high）
mypy src/autovision/errors.py src/autovision/runtime/env.py \
    src/autovision/runtime/checkpoint.py src/autovision/runtime/fsdp.py \
    src/autovision/config                # 核心模块类型门禁
pytest tests -q                          # 测试（含金标准快照）
python scripts/smoke_enterprise.py --json smoke-report.json  # 企业冒烟
# CI 另有 core coverage gate（19 个文件，--cov-fail-under=30）
python scripts/check_provenance.py       # 溯源门禁（禁词扫描 + allowlist 卫生）
python scripts/check_layering.py         # 分层契约（单向依赖）
python scripts/check_docs_links.py       # 公开树断链检查（链接 + 反引号仓库路径）
scripts/gen_model_zoo.py --check         # 模型库页面与注册表一致性（pytest 同款）
mkdocs build --strict                    # 文档站点构建（零警告）
```

## 测试规范

- 新功能必须带测试。数值 / 行为类改动**优先对拍**：与官方实现比对
  （bit-equal 优先）、与独立构造的 naive oracle 比对、或断言数值容差——
  而不是只断言「跑通」。
- 金标准快照在 `tests/golden/`：**有意的**配方变更后删除快照重新录制
  （对应测试跑两遍）；快照更新必须在本 PR 说明原因。
- 需要网络 / GPU 的测试一律用门控环境变量（`AUTOVISION_HF_TESTS=1`、
  `AUTOVISION_DDP_TESTS=1`）；默认路径必须 CPU-only 可跑、CI 无网络依赖。
- 修复 bug 时先写能复现的测试，再修实现。

## 文档规范

- 站点页面放在 `docs/` 下，**必须**在 `mkdocs.yml` 导航注册（`--strict`
  构建要求）；内部分析类文档用 `exclude_docs` 显式排除，不静默发布。
- 相对链接必须指向存在的文件（`scripts/check_docs_links.py` 强制；站点内
  链接由 `mkdocs build --strict` 强制）。
- 数字纪律：性能 / 精度数字只进 `docs/benchmarks.md`，并遵循该文首的不可变
  协议——对手数字只引官方发表值并给出处，自家数字必附复现脚本，只增不改、
  修正进勘误节。
- 模型库页面（`docs/model_zoo.md`）由注册表机器生成，勿手改；注册表变更后
  运行 `scripts/gen_model_zoo.py` 重新生成。
- **国际化**：中文是默认语言（`docs/*.md`）；英文副本使用同名 `*.en.md`
  （如 `index.en.md`）。未提供英文副本的页面由 `mkdocs-static-i18n`
  自动回退到中文；新增或修改页面时，若已有对应的 `.en.md`，请同步更新。
- README 语言切换：`README.md`（中文默认）与 `README.en.md` 顶部互链；
  英文站构建在 `/en/`，中文站保持在 `/`。

## 许可与溯源红线

- 每个新增源文件带 Apache-2.0 版权头（样式见任一现有文件首行）。
- 新依赖必须先过许可审查。**AGPL 类只允许 extras 隔离**（模板见
  `pyproject.toml` 的 `ultralytics` 条目）：不得进入核心 `dependencies`、
  不得在模块级 import、不得成为默认安装路径的一部分。
- 权重资产必须钉 `revision` + `sha256` + 许可三元组；受限许可
  （CC-BY-NC、自定义 License）必须默认不下载、要求显式自证接受。
- 二进制权重不进仓（`.gitignore` 已挡 `*.pt / *.pth / *.onnx / *.engine`）；
  第三方代码 / 模型入库需登记 `NOTICE` 并保持字节级可溯源（重新打包的容器
  必须附逐位忠实性对拍）。
- 不允许删除或弱化既有门禁来让变更通过评审。

## 提交与 PR

- 提交信息：一行摘要写明「做了什么 + 关键验证读数」，正文写动机、方案与
  验证方式（对拍读数、门禁结果）。参考既有提交历史。
- PR 使用 `.github/PULL_REQUEST_TEMPLATE.md` 清单；CI 全绿 + 至少一轮评审。
- 面向用户的可见变更记入 `CHANGELOG.md` 的 `[Unreleased]` 段（格式遵循
  Keep a Changelog；版本政策见 `docs/release_process.md`）。
- 公共 API（`autovision.*` 顶层导出、CLI 命令 / 参数、配置字段）的弃用遵循
  发布流程中的弃用政策：先标注、至少跨一个 MINOR 才移除。

## 问题报告

用仓库的 issue 模板（bug 报告 / 功能请求）；许可与合规问题请按
`NOTICE` 的联系方式处理，不要在公开 issue 中粘贴受限权重的直链。
