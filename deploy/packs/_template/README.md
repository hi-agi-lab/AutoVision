# Hardware pack template

Copy this directory to `deploy/packs/<vendor>/` for one customer/chip.

1. Export ONNX + postprocess spec from AutoVision.
2. Run `convert.py` with the vendor SDK command; it validates inputs, copies
   `postprocess.json`, writes `pack.json`, and executes the command in the
   pack directory.
3. Run `parity.py` between reference and target JSONL on the target device
   (20 images minimum). A pixel-tolerance failure blocks delivery.
4. Record SDK/runtime version, quantization, input size, batch, latency
   (p50/p90), memory, and license in `benchmark.md`.

Inputs:

```bash
python convert.py --onnx model.onnx --postprocess model.onnx.postprocess.json \
    --out deploy/packs/rknn --vendor rknn \
    --vendor-command "rknn-toolkit2 convert --onnx model.onnx --out model.rknn"
```

Parity:

```bash
python parity.py --reference reference.jsonl --candidate target.jsonl \
    --box-tolerance 1.0
```

The vendor SDK is not vendored in this repository; its license is recorded
separately. This is a delivery contract, not a universal backend matrix.
