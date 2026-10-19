# PyPI 发布指南

本文是 AutoVision 的端到端 PyPI 发布手册：一次性账号配置、版本切分、构建、TestPyPI
演练、正式 PyPI 上传、CI trusted publishing 与故障排查。

> **发行名 ≠ import 名**：PyPI 发行名是 `autovision-framework`；import 包名仍是
> `autovision`；命令行仍是 `autovision-cli`。
>
> PyPI / TestPyPI 上的 `autovision` 已被无关项目占用；`AutoVision` 归一化后等于
> `autovision`，同样不能用。**不要**使用 `autovision` 或 `auto-vision` 作为发行名。

## 0. 当前状态

| 项目 | 状态 |
|---|---|
| PyPI 发行名 | `autovision-framework`（PyPI / TestPyPI 当前均可用） |
| import 名 | `autovision` |
| CLI | `autovision-cli` |
| 版本源 | `src/autovision/_version.py`（当前 `0.3.0`） |
| 已具备 | `pyproject.toml` 元数据、`README_PYPI.md`、`MANIFEST.in`、`uv.lock`、CI `publish` job、Pages 部署 workflow |
| 待配置 | 公开仓库与文档 URL（用 `scripts/configure_release.py`）、PyPI trusted publisher、正式版本 tag |
| **许可状态** | DEIMv2 派生的 `SpatialPriorModule` 已删除；0.3.0 可按 Apache-2.0 发布，见 `THIRD_PARTY_LICENSES.md` |

> ✅ **许可阻塞已解除（2026-10-17）**：DEIMv2 派生的 `SpatialPriorModule`
> 已删除，DINOv3 ViT 改用 no-STA adapter。发布前请重新构建 wheel/sdist
> （旧产物仍含 STA），再执行 `twine check` 与许可审计。

包**尚未上传** PyPI；本文中的 `pip install autovision-framework` 要等第一次发布后才生效。

## 1. 一次性配置

### 1.1 账号与 2FA

- 注册 <https://pypi.org> 与 <https://test.pypi.org> 账号（两个独立账号体系）。
- 两个账号都开启 2FA；PyPI 强制要求。
- 记录账号邮箱，后续 token / trusted publisher 配置用得到。

### 1.2 方式 A：API token（手动上传）

1. PyPI → Account settings → API tokens → Add API token。
2. 首次发布前 token 只能选 **Entire account**；项目创建后可改为仅该项目。
3. TestPyPI 单独创建 token，不要混用。
4. 只在当前 shell 使用，不要写进仓库：

```bash
export TEST_PYPI_TOKEN=pypi-...
export PYPI_TOKEN=pypi-...
```

### 1.3 方式 B：Trusted Publishing（推荐，CI 无长期 token）

PyPI → 项目页面 → Publishing → **Add a pending publisher**，填写：

| 字段 | 值 |
|---|---|
| PyPI Project Name | `autovision-framework` |
| Owner / Repository | 实际公开仓库（如 `ORG/AutoVision`） |
| Workflow name | `ci.yml` |
| Environment name | `pypi` |

TestPyPI 同样配置一份 pending publisher，environment 用 `testpypi`（如果后续加
TestPyPI workflow）。仓库的 `.github/workflows/ci.yml` 已有 tag 触发的 `publish`
job，使用 `pypa/gh-action-pypi-publish` 与 OIDC，不需要 token。

> pending publisher 必须在第一次上传前配置好；否则 tag 推上去后 `publish` job 会以
> `invalid-publisher` 失败。

### 1.4 公开 URL 与 project.urls

首次发布前必须确定：

- 公开仓库地址（GitHub / GitLab）
- 文档站地址（GitHub Pages / GitLab Pages）
- Issue 地址

用仓库内的脚本一次性写入 `pyproject.toml` 与 `mkdocs.yml`：

```bash
python scripts/configure_release.py \
    --repo-url https://github.com/ORG/AutoVision \
    --docs-url https://ORG.github.io/AutoVision/
python scripts/configure_release.py --check
```

脚本会写入 Homepage / Documentation / Repository / Issues / Changelog，并
设置 `site_url` / `repo_url`；`--check` 在 URL 仍是占位值或缺字段时退出 1。
GitHub Pages 部署由 `.github/workflows/docs.yml` 在 `main` push 时执行
（需在仓库 Settings → Pages 选择 GitHub Actions）。

PyPI 页面正文来自 `README_PYPI.md`。该文件必须**自包含**：不要出现
`docs/...`、`CONTRIBUTING.md` 之类的相对链接；PyPI 不从仓库根解析。若要在 PyPI
页面链接文档，使用绝对 URL。

### 1.5 确认发行名仍可用

发布前重新确认一次（404 表示可用）：

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  https://pypi.org/pypi/autovision-framework/json
curl -s -o /dev/null -w "%{http_code}\n" \
  https://test.pypi.org/pypi/autovision-framework/json
```

名字一旦在 PyPI 创建就**不能改名**；确认后再投入发布。

## 2. 版本与发布准备

### 2.1 版本规则

- 单一版本源：`src/autovision/_version.py`。`pyproject.toml` 通过
  `[tool.setuptools.dynamic]` 读取，运行时的 `autovision.__version__` 也来自这里。
- 语义化版本：MAJOR 破坏性变更、MINOR 新能力、PATCH 修复。
- 发布 tag 统一 `vX.Y.Z`，必须与 `_version.py`、CHANGELOG 一致。
- 当前 `0.3.0`，CHANGELOG 的 0.3.0 段已收编，`[Unreleased]` 保留为空段；
  发布时打 `v0.3.0`。

### 2.2 切版本

1. 确认 `src/autovision/_version.py` 为 `0.3.0`（下一次发布时改成新的
   `X.Y.Z`）：

   ```python
   __version__ = "0.3.0"
   ```

2. `CHANGELOG.md` 的 `[0.3.0]` 段已收编，`[Unreleased]` 为空段；下一次发布
   时把新的 `[Unreleased]` 收编为新的 `[X.Y.Z] - YYYY-MM-DD`，并补空段。
3. 确认 `docs/release_process.md` 的版本检查使用新发行名：

   ```bash
   python -c "from importlib.metadata import version; import autovision; \
   assert version('autovision-framework') == autovision.__version__"
   ```

4. 重新安装 editable 包，更新本地 metadata：

   ```bash
   pip install -e . --no-deps
   ```

5. commit 后打 tag（tag 必须在发布提交上）：

   ```bash
   git tag v0.3.0
   git push origin main --tags
   ```

### 2.3 依赖与锁文件

- 发行名改动后运行：

  ```bash
  uv lock
  uv lock --check
  ```

- `super-gradients` 与 `export` / `transformers` / `rfdetr` / `video` 在
  `[tool.uv] conflicts` 中声明为互斥（上游钉死 onnxruntime / protobuf / numpy
  版本）。这是预期行为，不要为了让它们共存而放宽版本。

### 2.4 PyPI README

- PyPI 页面使用 `README_PYPI.md`，`pyproject.toml` 中
  `readme = "README_PYPI.md"`。
- 保持短、自包含：安装、quickstart、常用 extras、许可。
- 不要使用 `<details>`；不要使用相对链接。
- 修改后重新构建，并用 `twine check` 验证渲染。

### 2.5 sdist / wheel 内容约定

- wheel 是纯 Python（`py3-none-any`）；可选 CUDA/C++ 源不进入 wheel。
- sdist 通过 `MANIFEST.in` 携带：
  - `gtr/csrc/*.cu`、`*.cuh`、`build.sh`、`setup.py`、`test_gla.py`
  - `_msda_src/*.cpp`、`*.cu`、`*.cuh`、`*.h`
- GTR 自定义 GLA 核未编译时，运行时会回退到 fla，功能可用但少约 2× 加速；
  需要自定义核的用户应从 sdist / 源码仓库构建。

## 3. 发布前门禁（本地）

在干净工作树上执行，全部通过再继续：

```bash
ruff check src tests scripts
bandit -r src/autovision -ll -ii -q
mypy src/autovision/errors.py src/autovision/runtime/env.py \
    src/autovision/runtime/checkpoint.py src/autovision/runtime/fsdp.py \
    src/autovision/config
pytest tests -q
python scripts/smoke_enterprise.py --json smoke-report.json

python scripts/check_provenance.py
python scripts/check_layering.py
python scripts/check_docs_links.py
python scripts/gen_model_zoo.py --check
mkdocs build --strict

uv lock --check
```

说明：

- `pytest tests -q` 约 20 分钟（视硬件）；CI 的 `test` job 会跑同一套。
- `gen_model_zoo.py --check` 防止模型库页面与注册表漂移。
- `mkdocs build --strict` 会拦截 nav 缺失、孤儿页面和站内坏链。
- 不要在 dirty tree 上发布；tag 必须指向发布提交。

## 4. 构建与本地验证

```bash
rm -rf dist
uv build
ls -lh dist
```

预期产物：

```text
dist/autovision_framework-X.Y.Z-py3-none-any.whl
dist/autovision_framework-X.Y.Z.tar.gz
```

检查元数据与内容：

```bash
uv tool run --from twine twine check dist/*

# wheel：不应出现 csrc；应包含 METADATA / LICENSE / NOTICE
python3 -m zipfile -l dist/autovision_framework-*.whl | \
  grep -E "csrc|METADATA|NOTICE|entry_points"

# sdist：应包含 GTR / MSDA 的 CUDA/C++ 源
tar tzf dist/autovision_framework-*.tar.gz | \
  grep -E "gtr/csrc/|_msda_src/"
```

检查关键元数据：

```bash
unzip -p dist/autovision_framework-*.whl \
  'autovision_framework-*.dist-info/METADATA' | sed -n '1,25p'
```

应看到：

- `Name: autovision-framework`
- `License-Expression: Apache-2.0`
- `License-File: LICENSE`、`License-File: NOTICE`
- `Requires-Python: >=3.10`

可以做一个不装依赖的布局冒烟：

```bash
rm -rf /tmp/av-wheel && mkdir -p /tmp/av-wheel
python3 -m zipfile -e dist/autovision_framework-*.whl /tmp/av-wheel
cd /tmp
PYTHONPATH=/tmp/av-wheel <带 torch 的 python> -c \
  "import autovision; print(autovision.__version__); \
   import autovision.metrics.map_evaluator; print('deep imports ok')"
```

真正的安装验证放到 TestPyPI（下一节），这样依赖也走真实安装链路。

## 5. TestPyPI 演练

TestPyPI 是独立的索引，和正式 PyPI 互不影响。**先在这里完整演练一遍。**

```bash
UV_PUBLISH_TOKEN="$TEST_PYPI_TOKEN" \
  uv publish --publish-url https://test.pypi.org/legacy/ dist/*
```

或者：

```bash
uv tool run --from twine twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  --username __token__ --password "$TEST_PYPI_TOKEN" dist/*
```

在干净虚拟环境里安装并验证：

```bash
uv venv /tmp/av-testpypi

uv pip install --python /tmp/av-testpypi/bin/python \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  autovision-framework==X.Y.Z

/tmp/av-testpypi/bin/python -c \
  "import autovision; print(autovision.__version__)"
/tmp/av-testpypi/bin/autovision-cli --help
```

说明：

- TestPyPI 的依赖不完整，必须加 `--extra-index-url https://pypi.org/simple`；
  这个组合仅用于测试，生产安装不要照抄。
- 检查 `pip list` / `uv pip list` 里是 `autovision-framework`，不是旧的
  `autovision`。
- 打开 TestPyPI 项目页，确认 README、许可、文件列表正常。
- TestPyPI 可以用于反复演练，但不要把它当正式版本来源。

## 6. 正式发布到 PyPI

### 6.1 手动 token 上传

```bash
UV_PUBLISH_TOKEN="$PYPI_TOKEN" uv publish dist/*
```

或：

```bash
uv tool run --from twine twine upload \
  --username __token__ --password "$PYPI_TOKEN" dist/*
```

### 6.2 tag + CI trusted publishing

确认 PyPI pending publisher 已配置后：

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

GitHub Actions 会执行：

- `lint` / `provenance` / `docs` / `test` 四道门禁；
- 全部通过后，`publish` job 用 `pypa/gh-action-pypi-publish` 上传 `dist/*`。

`publish` job 定义在 `.github/workflows/ci.yml`，
`if: startsWith(github.ref, 'refs/tags/v')`，environment 为 `pypi`。

### 6.3 发布后验证

```bash
pip index versions autovision-framework

uv venv /tmp/av-pypi
uv pip install --python /tmp/av-pypi/bin/python autovision-framework
/tmp/av-pypi/bin/python -c \
  "import autovision; print(autovision.__version__)"
/tmp/av-pypi/bin/autovision-cli --help
uv pip show --python /tmp/av-pypi/bin/python autovision-framework
```

检查 PyPI 项目页：

- 项目名、summary、README、许可正确；
- Source / Documentation / Issues 链接正确（`[project.urls]`）；
- 文件列表包含 wheel 与 sdist；
- `pip install autovision` 不会安装本项目（安装说明要写清发行名）。

### 6.4 版本不可覆盖

PyPI 不允许覆盖已上传版本：

- 小修复：升 PATCH，重新构建上传；
- 严重问题：在 PyPI 页面 **Yank** 该版本，再发 PATCH；
- 不要试图复用版本号或强制覆盖。

## 7. 发布后收尾

- 更新 README / 文档站的安装命令为 PyPI 安装（`pip install autovision-framework`）。
- 更新 CHANGELOG 与 release note。
- 确认 tag 指向发布提交，且 `vX.Y.Z` 与 `_version.py` 一致。
- 在干净环境做一次最终 smoke：
  `import autovision`、`autovision-cli --help`、`ensure_coco8()` 训练 3 步。
- 若使用 `uv`：`uv lock` 后提交更新后的 `uv.lock`。
- 记录发布 URL、commit、tag 与上传时间，便于回滚/审计。

## 8. 故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 上传时报 name not available / too similar | 与已有 `autovision` 太像 | 使用 `autovision-framework`；不要用 `auto-vision` 或 `AutoVision` |
| `403 Invalid or non-existent authentication` | token 错误 / 2FA 未开 / 作用域不对 | 重新创建 token；确认 `__token__` 用户名 |
| `invalid-publisher` | workflow 文件名、仓库、environment 与 pending publisher 不一致 | 按 PyPI 配置逐项核对 `ci.yml` / `pypi` |
| `File already exists` | 版本号已上传 | 升版本；不要覆盖 |
| PyPI 页面 README 链接坏 | 使用了相对链接 | 改用 `README_PYPI.md` + 绝对 URL |
| 缺少 LICENSE / NOTICE | `license-files` 配置丢失 | 确认 `license-files = ["LICENSE", "NOTICE"]` 后重建 |
| wheel 缺模块 | namespace 包 / packages.find 配置错误 | 对比 wheel 与源码 `.py` 数量；检查 `exclude` |
| GTR 自定义核不可用 | wheel 不带 `.so` | 正常回退 fla；需要自定义核时从 sdist 构建 |
| TestPyPI 安装依赖失败 | TestPyPI 缺依赖 | 加 `--extra-index-url https://pypi.org/simple`（仅测试） |
| `uv lock` 报 extras 冲突 | super-gradients 与 export/transformers/rfdetr/video 冲突 | 保持互斥声明；不要同时安装 |

## 9. 本次仓库的速查

关键文件：

- `pyproject.toml`：发行名、元数据、PEP 639 许可、packages.find
- `README_PYPI.md`：PyPI 页面正文
- `MANIFEST.in`：sdist 的 CUDA/C++ 源
- `src/autovision/_version.py`：版本单一源
- `CHANGELOG.md`：版本段
- `.github/workflows/ci.yml`：`publish` job
- `docs/release_process.md`：发布流程总览

发布前必须补齐：

1. `[project.urls]` 与公开仓库/文档 URL；
2. PyPI trusted publisher（或 API token）；
3. 版本号、CHANGELOG、tag 三者一致。

最小命令序列：

```bash
UV_DEFAULT_INDEX=https://pypi.org/simple uv lock && uv lock --check
# 必须匹配不到（返回 1 才继续）；若输出内部地址则停止发布：
! grep -q agentsmirror uv.lock
rm -rf dist && uv build
uv tool run --from twine twine check dist/*
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN" dist/*
# 在干净环境验证 TestPyPI 安装后：
uv publish --token "$PYPI_TOKEN" dist/*
```

## 10. 发布检查清单

- [ ] 发行名 `autovision-framework` 仍可用
- [ ] `[project.urls]` 已填公开 URL
- [ ] `README_PYPI.md` 无相对链接
- [ ] `_version.py` / CHANGELOG / tag 一致
- [ ] `uv lock --check` 通过
- [ ] `uv.lock` 使用公共 PyPI（无内部镜像地址）
- [ ] 全量测试与 CI 全套门禁通过（lint / provenance / docs / test + smoke / coverage）
- [ ] `uv build` 成功
- [ ] `twine check` 通过
- [ ] wheel 不含 csrc，sdist 含 GTR/MSDA 源
- [ ] TestPyPI 上传并安装验证通过
- [ ] PyPI 正式上传
- [ ] PyPI 页面 / 安装 / CLI smoke 通过
- [ ] README、文档、CHANGELOG、uv.lock 已收尾
