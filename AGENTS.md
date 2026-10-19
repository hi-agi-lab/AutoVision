# AGENTS.md — AutoVision 贡献与代理指南

本文件给 AI 代理和新贡献者一份可执行的仓库地图与硬约束。
与 `CONTRIBUTING.md` / `docs/release_process.md` 冲突时，以仓库正式文档为准。

## 1. 先读什么

- 架构总览：`docs/framework_design.md`
- 发布/弃用：`docs/release_process.md`、`docs/lts_policy.md`
- 合规/溯源：`NOTICE`、`THIRD_PARTY_LICENSES.md`、`scripts/check_provenance.py`
- 已知边界：`docs/known_limits.md`

## 2. 仓库地图

```
src/autovision/
├── cli/                 # 30 条命令；参数校验与 Python API 同源
├── config/              # Pydantic 配置与校验
├── data/                # 数据集、转换、mmap、interchange
├── engine/              # 训练循环（step / DDP / AMP / checkpoint）
├── export/              # ONNX / TensorRT / OpenVINO + parity 门禁
├── inference/           # 参考推理运行时（engine / stream / multi_stream / results）
├── metrics/             # mAP、误差分析、混淆矩阵、任务指标
├── registry/            # 模型/权重注册表 + model card
├── runtime/             # 环境、checkpoint、FSDP、日志
├── serving/             # FastAPI 参考 serving
├── tasks/               # 8 个任务面；TaskNetwork / TrainModel
├── tracking/            # ByteTrack + LineZone/PolygonZone/TrackSmoother
├── visualize/           # 任务感知 annotator + 区域叠加
└── augment/             # 检测/pose 增广
tests/                   # 与 src 对应的 pytest 测试
docs/                    # MkDocs（build --strict）
scripts/                 # 发布、溯源、基准、数据工具
```

## 3. 硬约束

### 3.1 推理结果契约

- `TaskNetwork.postprocess` 返回 **dict**，键名以 `src/autovision/keys.py`
  为准（`bboxes` / `scores` / `labels` / `masks` / `keypoints` / `polygons` /
  `mask` / `depth` / `segment_ids`）。不要在业务代码里新增裸字符串键。
- 消费侧（可视化、serving、stream、误差分析）优先使用
  `autovision.results.InferenceResult`：它负责校验、筛选、
  `to_numpy()`、`to_record()` 与 `to_json(masks="rle"|"dense"|"none")`。
- 不要把 `InferenceResult` 塞回训练/导出热路径；训练仍以 tensor/dict 为准。
- 分割 mask 跨 API 边界默认 RLE（`pycocotools.mask`）；`(N,H,W)` dense mask 不得
  直接 `tolist()` 进 HTTP/JSONL。大数组必须走 `max_elements` 摘要。

### 3.2 可选依赖

- `torch` / `torchvision` 是核心依赖；`cv2` 只在 `video` extra，`fastapi` 只在
  `serving` extra，`transformers` / `timm` / `ultralytics` 等按 extra 惰性导入。
- 新可选依赖必须在函数内部导入并给出 `pip install -e '.[extra]'` 的明确报错。
- 不能在模块顶层导入 `cv2`、`fastapi`、`transformers` 等非核心包。
- `ultralytics` 是 AGPL：只允许在隔离的 encoder 模块中惰性导入，绝不进核心路径。

### 3.3 数值与导出

- 任何影响前向/解码/导出数值的改动，必须跑对应 parity / 金标准测试：
  `tests/test_e2e_parity.py`、`tests/test_deployment.py`、`tests/golden/`。
- 导出路径默认 `verify=True`；不得为了跑通而关闭对拍门禁。
- 新增任务/变体必须同时提供：权重加载、`verify_onnx_export_outputs`、
  一个最小端到端测试。

### 3.4 弃用

- 公共 API（`autovision.*` 顶层导出、CLI 命令/参数、配置字段）先标注 +
  运行时警告，至少跨一个 MINOR 才移除；LTS 分支不引入弃用。
- 警告文案必须写明 `deprecated in X.Y.Z` 与 `remove in A.B.C`。
- 不要用“静默别名”长期保留两个名字；参考 `supervision` 的教训：
  重复入口会让维护面翻倍。

### 3.5 合规/溯源

- 新代码不得包含上游仓库的受版权注释或大段逐字复制；算法参考写进
  `THIRD_PARTY_LICENSES.md` / provenance ledger。
- 受限许可权重默认不下载；新增资产必须登记许可证与 sha256。
- `scripts/check_provenance.py`、`scripts/check_layering.py` 必须保持绿色。

## 4. 代码风格

- 类型提示必备；新模块应能通过：
  `.venv/bin/mypy --config-file pyproject.toml <files>`
- 格式/静态检查：`.venv/bin/ruff check src tests scripts`
- 矢量化为先；热路径不要写 `for det in detections`。
- 注释只解释“为什么/约束”，不复述代码；函数级说明放 docstring。
- 新函数/类必须有简洁 docstring；公共 API 用 Google 风格参数说明。
- 测试：AAA 结构；参数化用 `pytest.param(..., id="slug")`；每个测试至少一行
  docstring 说明场景。

## 5. 常用命令

```bash
# 目标测试（快）
.venv/bin/python -m pytest tests/test_inference_results.py tests/test_visualize.py \
    tests/test_analytics.py tests/test_confusion.py tests/test_mask_serialization.py -q

# 推理/流/服务
.venv/bin/python -m pytest tests/test_inference_runtime.py tests/test_stream_pipeline.py \
    tests/test_multi_stream.py tests/test_tracking.py tests/test_serving.py -q

# 转换/误差/SAHI
.venv/bin/python -m pytest tests/test_convert_extended.py tests/test_interchange.py \
    tests/test_error_analysis.py tests/test_deployment.py tests/test_sahi_merge.py -q

# 静态检查
.venv/bin/ruff check src tests scripts
.venv/bin/mypy --config-file pyproject.toml src/autovision/results.py \
    src/autovision/inference/stream.py src/autovision/visualize/annotators.py \
    src/autovision/tracking/analytics.py src/autovision/metrics/confusion.py

# 文档
mkdocs build --strict
```

## 6. 修改流程

1. 先复现/定位：写一个最小失败用例或指出具体文件行。
2. 最小改动 + 测试；不要顺手重构无关模块。
3. 跑目标测试 + 相关回归；记录失败是既有还是新增。
4. 更新 `CHANGELOG.md`（功能性变更/行为变更/安全修复）。
5. 需要时更新 `docs/` 与 README；`mkdocs build --strict` 必须零警告。
6. 不改未提交的他人改动；发现脏工作区先确认归属。

## 7. 代理易错点

- **不要**把 `InferenceResult` 当训练容器；它的字段是 numpy/torch 混合的边界类型。
- **不要**在 `visualize` 里假设 `bboxes` 一定存在；pose/OBB/分类/语义/深度结果
  走 `annotate_result` 的任务分派。
- **不要**默认 dense mask 可 JSON 序列化；用 `to_json(masks=...)`。
- **不要**在 `run_stream` 主循环之外更新 tracker/zone 状态；它们是按帧有序的状态机。
- **不要**把 `LineZone`/`PolygonZone` 用在 `track=False` 的流上；运行时会报错。
- **不要**在 tile 合并前丢弃 tile 间重复框；`combine_object_detection_tiles`
  现在统一做 per-label 过滤。
- **不要**用 `xml.etree.ElementTree` 解析不可信标注；统一 `defusedxml`。
