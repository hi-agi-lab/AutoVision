# Scripts

本目录只保留发布、用户功能和可复现工具。按用途分组：

## 发布门禁与元数据

| 脚本 | 用途 | 引用位置 |
|---|---|---|
| `check_provenance.py` | 溯源/禁词扫描与 allowlist 卫生 | CI、CONTRIBUTING、发布手册 |
| `check_layering.py` | `autovision` 单向分层契约 | CI、CONTRIBUTING、发布手册 |
| `check_docs_links.py` | 公开树相对链接/反引号路径检查 | CI、CONTRIBUTING、发布手册 |
| `gen_model_zoo.py` | 从注册表生成 `docs/model_zoo.md`；`--check` 防漂移 | CI、测试、发布手册 |
| `configure_release.py` | 写入仓库/文档站公开 URL；`--check` 校验 | 发布/PyPI 文档 |
| `publish_clean_cut.sh` | 生成并校验单提交发布快照（tree 相等、无 parent）；默认只建本地分支/tag，`--push` 才推送 | 发布手册、pre-push 钩子 |

## 权重、模型与功能

| 脚本 | 用途 |
|---|---|
| `download_gtr_assets.py` | 从 HF `Phoenix8125/GTR` 下载并校验 GTR 官方权重 |
| `train_spectrum.py` | 训练 `detr/n` / `detr/s` 产品档位 |
| `eval_spectrum_coco.py` | 在 COCO split 上评测导出的档位模型 |
| `measure_spectrum_benchmarks.py` | 复现基准页的参数量/FLOPs/延迟 |
| `measure_new_backends.py` | 复现 torch / ONNX / OpenVINO / TensorRT 后端读数 |
| `measure_int8_accuracy.py` | fp32 vs int8 全量 AP 报告 |
| `eval_gtr_pose_ap.py` | 复现 GTR pose 官方协议 AP |
| `self_train_detect.py` | 伪标注 / 多轮自训练闭环 |
| `import_eomt_checkpoint.py` | 导入第三方 DINOv3 EoMT 检查点 |
| `prepare_apache_detector_weights.py` | 维护者工具：构建 Apache-2.0 检测器权重库 |

## 实验性脚本（不随发布承诺）

`experimental/` 下的脚本不进入 CI、不承诺 API 兼容，也不对应用户文档中的稳定路径。
当前包括 TFLite 转换实验与已阻塞的 COCO warm-start 构建骨架；状态见各自文件头。

## 历史研究脚本

VisDrone 训练实验和用户自持权重的本地可视化脚本不属于发布树；它们依赖本机数据集/
私有权重路径，已移出仓库（git 历史仍保留）。需要时请从历史提交恢复，或放在仓库外的
研究目录。
