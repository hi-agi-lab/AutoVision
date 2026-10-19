# Known limits and roadmap

> This page collects externally visible boundaries and engineering notes.
> Export-specific details are in [export & deployment](export.md).

## Positioning

A training + reference inference framework. Training, evaluation and export are
production paths; tracking and serving are reference implementations. Serving
supports an optional API key (`AUTOVISION_SERVING_API_KEY` or `api_key=`),
request-body/pixel limits, thread-pool inference, bounded queuing
(`queue_timeout_s`, returning 503 when saturated), a per-call deadline
(`inference_timeout_s`, default 120s; a timeout returns 504, marks the instance
unresponsive, and from then on `/readyz` returns 503 with `poisoned=true` while
`/healthz` carries a `poisoned` field — orchestrators should drain traffic and
restart the process), `/readyz`, TLS, multiple workers,
in-process token-bucket rate limiting, prediction-distribution drift (PSI,
`/drift`) and A/B/canary routing; `/docs` and `/openapi.json` are disabled by
default. Global quotas, alerting, cross-process consistent rate limiting,
A/B statistical significance and automatic canary rollback remain the
responsibility of a production gateway.

## Runtime and integrity

- **Determinism**: `AUTOVISION_DETERMINISTIC=1` enables torch deterministic
  operators and cuDNN deterministic mode (some operators have no deterministic
  implementation and will error; it is usually slower). Disabled by default.
- **fp16**: `precision="16-mixed"` uses `GradScaler` on CUDA/NPU and saves/restores
  its state with the checkpoint; bf16/fp32 do not need it. AMP overflow is skipped
  by the scaler (with scale backoff), not aborted by the finite-gradient check.
- **Resume**: checkpoints restore Python/NumPy/Torch RNG state, but not the
  mid-epoch sampler position, and they do not validate changed config/version;
  optimizer state restores the checkpoint's `lr` and related values. Resume from
  an epoch boundary when strict trajectory reproduction is required. Loading a
  checkpoint written by a different AutoVision version logs a warning (not an
  error). Checkpoint/export/manifest writers first sweep crash-orphaned temp
  files older than one hour for the same target, then start their own write.
- **mmap cache integrity**: cache files are validated on open; a corrupt or
  truncated file fails with a rebuild hint (delete it, or rebuild via
  `overwrite=True`). External truncation while the file is mapped (an outside
  writer sharing `AUTOVISION_DATA_CACHE_DIR`) crashes the process with SIGBUS —
  an OS-level constraint; never let untrusted processes write the cache dir.
- **Download integrity**: weight downloads verify pinned sha256 digests; dataset
  downloads write a temporary file and atomically replace it, and cached archives
  are re-verified for size/sha256 (when pinned) before extraction. Built-in
  COCO/VOC dataset entries have no official digest and only warn by default.
  `AUTOVISION_REQUIRE_DATASET_HASHES=1` rejects unpinned dataset downloads;
  `AUTOVISION_OFFLINE=1` rejects both dataset and weight downloads.
- **Legacy teacher checkpoints**: current exported task models load through the
  pickle-free loader; old exports embedding `__class__` or similar objects are
  rejected by default. Only trusted local artifacts may set
  `AUTOVISION_TRUST_TEACHER_CHECKPOINT=1` to enable the full unpickler (RCE risk
  is accepted by the caller).
- **mmap cache key**: includes label-file paths (COCO JSON / YOLO `.txt`), sizes,
  mtimes and the `ignore_classes` fingerprint; editing labels or changing ignored
  classes rebuilds the cache, and `overwrite=True` forces a rebuild. User-held
  weight resolution verifies sha256 (digest cached by path/size/mtime).
- **EoMT weights**: `load_weights=True` requires DINOv2/DINOv3 license opt-in. Hub
  checkpoints without a sha256 pinned in `WEIGHT_MANIFEST` are rejected unless
  `AUTOVISION_ALLOW_UNPINNED_WEIGHTS=1` is set (trusted mirror). The class head is
  rebuilt for the target class/query count; EoMT can be exported to ONNX through
  the legacy TorchScript path (dynamic batch disabled).
- **Visualization/video tools**: the box/mask/polygon helpers in
  `visualize/effects.py`, `geometry/boxes.py` and `geometry/masks.py` depend only
  on NumPy/Pillow; `mask_to_polygons` lazily depends on OpenCV (`video` extra).
  `process_video` / `VideoSink` / `ImageSink` are lightweight synchronous,
  RGB-convention pipelines and do not replace `run_stream`'s reader/infer/writer
  pipeline, backend switching or JSONL persistence.
- **Run manifests**: every training run writes `run_manifest.json` (command, git
  commit, environment allowlist, dependency versions, GPU); resume writes a
  separate `run_manifest.resume-*.json` without overwriting the original.
- **Early stopping/callbacks**: task training supports
  `early_stopping_patience` / `early_stopping_min_delta` (CLI `trainer_args`) and
  Python callback hooks; pretraining still runs the configured epochs.

## Enterprise boundaries

- **Training parallelism**: DDP (torchrun), FSDP task training
  (`trainer_args={"parallel": "fsdp"}`, full-state checkpoint/resume/export,
  tested with 2 ranks on one GPU) and gradient checkpointing are implemented.
  FSDP EMA updates gather full parameters according to
  `AUTOVISION_FSDP_EMA_INTERVAL`; the entire model is not auto-wrapped.
  DeepSpeed / ZeRO / elastic training are not integrated; multi-node NCCL and
  fault injection are not covered by public CI. See
  [parallel training](parallelism.md).
- **Registry**: file-system and MLflow backends both run the standard deployment
  gates (`gates=None` is fail-closed by default); rollback/audit remain
  file-system-only.
- **CI**: public CI runs CPU jobs on Ubuntu with Python 3.10/3.11; the test job
  installs the `serving`/`video`/`transformers` test dependencies, a core-subset
  coverage floor, bandit (medium/high), core-module mypy and the
  `scripts/smoke_enterprise.py` smoke gate. TensorRT/OpenVINO/multi-node DDP tests
  are gated by environment variables. Real TensorRT engine/parity was verified on
  RTX 4090D + TensorRT 11.4, including the int8 QDQ path and
  `export_tensorrt(verify=True)`; the public CI/nightly self-hosted workflow
  (`.github/workflows/trt-nightly.yml`) continues to cover it.
- **serving**: TLS, multiple workers, in-process rate limiting, drift and
  A/B/canary routing are available; `/metrics` and `/drift` also require auth
  when an API key is configured (K8s probes stay public). Rate-limit buckets and
  drift windows are per worker (cross-worker aggregation requires a metrics
  backend/gateway); `/drift` is service-level (mixed A/B traffic mixes
  distributions). There is no model hot reload (use rolling restarts);
  `deploy/k8s/` provides reference Deployment/Service/HPA/PDB/PVC manifests.
  Global quotas, alerting, A/B significance and automatic rollback remain
  platform responsibilities (see Positioning above).

## Licensing and STA migration

- **DEIMv2 components removed (2026-10-17)**: `SpatialPriorModule` was removed
  from the codebase; DINOv3 ViT variants remain on a no-STA adapter. Releases no
  longer contain DEIMv2 code and can be published as Apache-2.0. See the
  [provenance ledger](provenance/LEDGER.md).
- **Legacy STA-era ViT checkpoints are unsupported**: any baseline `.pt` or
  user-held ViT asset containing `backbone.sta.*` (`dinov3-vitt16` /
  `dinov3-vitt16plus` / `dinov3-vits16`) is rejected explicitly; name resolvers
  fail before reading/hashing the container. Retrain/export a no-STA checkpoint.
  DINOv3 ConvNeXt and the four user-held ConvNeXt assets are unaffected;
  `USER_HELD_LOADABLE_ASSETS` contains only those four, while the three ViTs stay
  in `USER_HELD_UNSUPPORTED_ASSETS` for audit only (`allow_unsupported=True`).
- **Restricted weights remain**: DINOv2 weights are CC-BY-NC 4.0, DINOv3 weights
  use the DINOv3 License, and user-held detectors are non-redistributable /
  non-commercial. These are weight terms unrelated to the removed STA code;
  external DEIMv2 weights remain non-commercial.

## Adaptation gaps

- The albumentations adapter is not integrated (transforms use torchvision v2).
- The super-gradients segmentation wrapper is not integrated
  (`CustomizableDetector` detection families are available).
- Hard-example weighting (`hard_examples=`) and long-tail sampling
  (`long_tail_sampler=`) are single-node only; DDP's DistributedSampler overrides
  sampler weights, so the combination is rejected explicitly.
- Label Studio `keypointlabels` are imported as detection boxes around
  `keypoint_radius`; pose training should use native keypoint formats (COCO
  keypoints / YOLO-pose).
- Label Studio rotated rectangles (`rotation != 0`) are imported as axis-aligned
  boxes with the rotation preserved in an annotation field; oriented training
  still uses YOLO-OBB / DOTA formats.
- Multi-stream runtime v1 does not perform cross-stream tracking/ReID; tracking
  is only in single-stream `run_stream`. Each stream's render writer is rebuilt
  when its resolution changes. `LineZone` / `PolygonZone` / `TrackSmoother` and
  the reader/infer/writer pipeline are also single-stream only; `run_multi_stream`
  keeps independent per-source batching.
- Serving's `max_image_pixels` applies per image, not to the total pixels or RLE
  response bytes in one request; high-concurrency large-image/dense-mask
  deployments need an additional gateway-level budget.

## Detection lineage status

`n` / `s` are spec-frozen (components and recipes are committed; pretrained
weights and COCO AP are pending release); `m` / `l` / `x` backbone directions are
fixed (timm ConvNeXt, tiny→m / small→l / base→x; selection criteria are
exportability + clean licensing) with implementation pending. `detr/gtrvit-s` is
planned. Only `released` tiers carry pretrained weights: the release gate requires
weights + model card (license / sha256 / seed) + full AP + reproducible latency.
Machine-readable status and variant lists are in the [model zoo](model_zoo.md).

## Export and deployment

- GTR models (including `detr/gtrvit-*`) cannot be exported to ONNX/TensorRT:
  the export entry point fails explicitly (the GLA custom CUDA kernel has no ONNX
  lowering; see `GTRViTBackbone.export_blocker`). The four GTR task models have
  export verify hooks ready and will take effect with the deployment work.
- The ONNX FP16 parity gate skips with a warning when no CUDA execution provider
  is available; ONNX example inputs follow the model device (GPU-trained models
  can be exported directly).
- Dynamic H/W support, the int8 first-layer fp32 recipe and TensorRT 11
  adaptation are covered in [export & deployment](export.md).
- The full int8 AP report (`scripts/measure_int8_accuracy.py`) requires a
  compatible ONNX Runtime / TensorRT environment; the `export` extra requires
  `numpy>=1.25` (onnx/onnxscript import prerequisite). Reinstall that extra to
  rerun. The current development venv still has an older numpy, so local ONNX
  integration tests are skipped by the environment gate.
- `examples/cpp/` is a reference runtime, not an ABI/SDK; the full build requires
  ONNX Runtime and OpenCV development libraries, while the host-only postprocess
  test needs only g++.
- GTR deployment work: the Triton GLA kernel is ported (2× speedup), but
  `torch.compile` is negative for the GLA segment, CUDA Graphs conflict with the
  custom kernel's in-place writes (requires op changes), and TensorRT deployment
  needs the official `gla_plugin.cu`.
- Official training augmentations (mosaic / mixup / zoom-out / color-jitter) are
  not ported; the official pose recipe uses `precision="32-true"`
  (`use_amp: False`).

## Offline and weights

- Distillation teachers request pretrained weights by default
  (`teacher_pretrained=True`); pass `teacher_pretrained=False` offline.
- Pretrained weight manifests are pinned by commit + LFS sha256
  (`src/autovision/encoders/weight_hashes.py`).
- Restricted weights (DINOv2 CC-BY-NC 4.0 / DINOv3 License) are not downloaded by
  default and require explicit acceptance; the user-held store is
  non-redistributable/non-commercial. See
  [user-held weights](user_held_weights.md).
