# 发布流程

本文档定义 AutoVision 的版本政策与发布清单。对应的变更记录见[变更日志](changelog.md)，
开发约定见[贡献指南](contributing.md)。

## 版本政策

- **语义化版本**（SemVer 2.0.0）：
  - `MAJOR`：公共 API 的破坏性变更（先经弃用政策过渡）；
  - `MINOR`：新能力、新任务面、新权重档位；
  - `PATCH`：向后兼容的修复。
- **版本号单一源**：`src/autovision/_version.py`。打包经 `pyproject.toml`
  动态读取（`[tool.setuptools.dynamic]`），运行时经 `autovision.__version__`
  暴露；三者不允许出现第二个书写位置。
- **发布 tag**：统一 `vX.Y.Z`。历史 tag `v0.2-clean` 为 0.2 线的干净谱系
  快照，保留原义，不重写、不移动。
- **弃用政策**：公共 API（`autovision.*` 顶层导出、CLI 命令与参数、配置
  字段）的弃用先在 MINOR 版本中标注（文档标注 + 运行时警告），至少跨一个
  MINOR 版本后才可移除；移除时在变更日志给出迁移方式。

## 发布清单（每次发布按序执行）

0. **公开 URL**：仓库/文档站地址确定后执行
   `python scripts/configure_release.py --repo-url ... --docs-url ...`，
   并用 `--check` 确认；GitHub Pages 由 `.github/workflows/docs.yml` 部署。
0.1 **许可审计（硬门禁）**：确认 `THIRD_PARTY_LICENSES.md` /
   `docs/provenance/LEDGER.md` 中无未清除的 BLOCKER。DEIMv2 派生的
   `SpatialPriorModule` 已于 2026-10-17 删除（ViT 改用 no-STA adapter），
   当前无发布阻塞；仍需确认构建产物不含 `backbone.sta.*` 代码路径。

1. **门禁全绿**（CI 同款，逐条执行）：

   ```bash
   ruff check src tests scripts
   bandit -r src/autovision -ll -ii -q
   mypy src/autovision/errors.py src/autovision/runtime/env.py \
       src/autovision/runtime/checkpoint.py src/autovision/runtime/fsdp.py \
       src/autovision/config
   pytest tests -q
   python scripts/smoke_enterprise.py --json smoke-report.json
   # CI 的 core coverage gate（19 个测试文件，--cov-fail-under=30）见 ci.yml
   python scripts/check_provenance.py
   python scripts/check_layering.py
   python scripts/check_docs_links.py
   scripts/gen_model_zoo.py --check
   mkdocs build --strict
   python scripts/configure_release.py --check
   # 锁文件必须只引用公共 PyPI；显式覆盖本机镜像配置。
   UV_DEFAULT_INDEX=https://pypi.org/simple uv lock --check
   ```

2. **版本号**：更新 `src/autovision/_version.py`；确认打包与运行时读到的
   版本一致：

   ```bash
   pip install -e . --no-deps
   python -c "from importlib.metadata import version; import autovision; \
   assert version('autovision-framework') == autovision.__version__"
   ```

3. **变更日志**：把 `CHANGELOG.md` 的 `[Unreleased]` 段收编为
   `[X.Y.Z] - YYYY-MM-DD`，并补一个空的 `[Unreleased]` 段。
4. **文档**：站点构建通过（`site/` 产物不进版本库）；模型库页面与注册表
   一致（`--check` 已覆盖）。
5. **发布快照**：对外仓库只承载发布快照——由当前树重建单提交快照分支，
   在快照提交上打 `vX.Y.Z` tag，再推送分支与 tag；开发线历史不进入对外仓库。使用
   `scripts/publish_clean_cut.sh` 生成并校验快照（tree 与当前提交一致、
   无 parent），默认只建本地分支/tag；确认后再加
   `--remote <publish> --push` 推送。**禁止**直接推送 `main` 或
   `v0.2-clean`（pre-push 钩子会拒绝）。
6. **发布后**：核对远程 tag 指向的树与本地发布树一致；tag 推送会触发 CI 在
   打 tag 的提交上执行套件（`v*` 触发规则），其中 tag ↔ 版本一致性检查
   （`pytest tests/test_infrastructure.py -k release_tag`）必须通过；下个开发
   周期继续在 `[Unreleased]` 段记录变更。

## PyPI 发布（发行名 ≠ import 名）

> 完整操作手册（账号、token、TestPyPI 演练、trusted publishing、故障排查）见
> [PyPI 发布指南](pypi_publish.md)。

- 发行名：`autovision-framework`；import 名：`autovision`；CLI：`autovision-cli`。
- PyPI 上 `autovision` 已被无关项目占用，**不要**使用该发行名；也不要把
  `AutoVision` 当作发行名（归一化后等于 `autovision`）。
- 先发 TestPyPI，再发 PyPI：

  ```bash
  uv build
  uv tool run --from twine twine check dist/*
  uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN" dist/*
  uv publish --token "$PYPI_TOKEN" dist/*
  ```

- tag 推送后由 CI 的 `publish` job 以 trusted publishing 上传；首次发布前在 PyPI
  配置 pending publisher（project `autovision-framework`、workflow `ci.yml`、
  environment `pypi`）。
- `README_PYPI.md` 是 PyPI 页面正文（自包含、无相对链接）；首次发布前补
  `[project.urls]` 与文档站绝对 URL。

## 权重与模型发布（政策）

> 完整操作手册（许可审计、模型卡、HuggingFace 上传、注册表、发布门禁、排错）见
> [模型发布指南](model_publish.md)。

- 仓库不携带权重二进制；注册表钉**身份**（sha256 / revision）而非分发
  （见[模型库](model_zoo.md)的资产类别：GTR 上游按需下载 / HF 清单 /
  可下载 warm-start / 用户自持）。
- GTR 官方权重（MIT）不随仓库分发；用 `scripts/download_gtr_assets.py` 从上游
  HF `Phoenix8125/GTR` 按需下载到 `AUTOVISION_GTR_ROOT`，逐文件核验 sha256；
  上游 gated 时先 `hf auth login`。自建镜像需保留上游 LICENSE/NOTICE 并登记
  revision + sha256。
- 自有训练产物如有对外发布需求，发布到 HuggingFace：
  1. 核对训练数据与权重的许可，确认允许再分发；
  2. 附 model card（训练口径、硬件、指标与复现脚本路径）；
  3. 把权重 sha256 回填到对应资产注册表；
  4. 在模型库页面登记（经 `scripts/gen_model_zoo.py` 重新生成）。
- 受限许可的权重（CC-BY-NC / 自定义 License）不允许进入公开注册表或随
  仓库 / 导出产物分发。
