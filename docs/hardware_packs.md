# 硬件伙伴包

目标：每个客户/芯片选型一个可交付包，而不是维护 15 个后端矩阵。
包目录约定：

```text
deploy/packs/<vendor>/
  README.md          # 芯片、SDK 版本、许可、限制
  convert.py         # ONNX + postprocess.json -> 厂商模型（调用 SDK）
  postprocess.json   # 与导出件同源的 decode 规格
  parity.py          # 目标端 JSONL 与参考 JSONL 对比
  parity_vectors/    # 图/期望输出（按项目提供）
  benchmark.md       # p50/p90、内存、功耗（如可得）
```

## 接口契约

1. 输入必须是阶段 2 的 ONNX + `<artifact>.postprocess.json`；
2. 目标端后处理必须按同一份规格（DETR top-k / PicoDet class-offset NMS）；
3. 目标端输出 JSONL 与参考实现同构（`predictions.bboxes/scores/labels`）；
4. `parity.py` 以像素容差与类别一致性判定（脚本可直接复用/改写）；
5. 记录目标 SDK 版本、量化方式、输入尺寸、batch、硬件与许可。

## 流程

```bash
# 模板：校验输入并准备 pack.json；真实转换由 --vendor-command 调 SDK
python deploy/packs/_template/convert.py \
    --onnx model.onnx --postprocess model.onnx.postprocess.json \
    --out packs/rknn --vendor rknn --vendor-command "rknn-toolkit2 convert ..."

# 目标端/参考端 JSONL 对比
python deploy/packs/_template/parity.py \
    --reference reference.jsonl --candidate target.jsonl --box-tolerance 1.0
```

模板本身不需要厂商 SDK；`_template/convert.py` 只做输入校验、复制 spec、
写 `pack.json`，并在提供 `--vendor-command` 时执行它。厂商 SDK 不随仓库分发。

## 边界

- 一个伙伴一个包；不做全厂商预研。
- 目标端 parity 归平台侧执行；仓库提供契约、模板与参考对比脚本。
- 许可：厂商 SDK/运行时的许可单独登记，不混入框架 Apache-2.0。
