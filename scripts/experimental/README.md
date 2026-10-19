# Experimental scripts

这些脚本不属于发布承诺，不进入 CI，也不保证 API/路径稳定：

- `convert_onnx_to_tflite.py`：ONNX → TFLite/LiteRT 边缘转换实验；工具链和 parity
  仍受已知阻断约束，详见文件头。
- `build_warm_start_coco.py`：COCO warm-start 构建骨架；当前 `build()` 明确抛出
  `NotImplementedError`，待实现后再迁回正式发布目录。

发布相关功能请使用上一级 `scripts/README.md` 中列出的稳定入口。
