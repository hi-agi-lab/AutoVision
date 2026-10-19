# User-held weights

> Seven third-party-trained COCO detectors (DINOv3 backbones + third-party
> detection heads) are repackaged for compliance and stored outside the
> repository; they are **not redistributable**. The asset table, licenses and
> sha256 values are in the [model zoo](model_zoo.md); this page covers the
> loading routes and required overrides. **Three ViT assets are STA-era and are
> not loadable today; the four ConvNeXt assets load normally.**

## Storage and licensing

- Default root: `~/Research/Model/AutoVision`; override with
  `AUTOVISION_USER_HELD_ROOT`.
- Each asset is a `weights.pt` + `warm_start.pt` + `ASSET.json` triplet.
- Asset names and both container sha256 values are pinned in
  `autovision.registry.user_held_assets` (`resolve_user_held_asset` resolves;
  `verify_user_held_asset` verifies offline).
- License: `DINOv3-License+NOASSERTION`. Backbones inherit DINOv3 License
  obligations; detection heads carry no license statement (all rights reserved by
  default). **No redistribution, no commercial use**; the assets are not shipped
  with the repository or exported artifacts and are not registered in
  `ASSET_MANIFEST`. Use constitutes acceptance of the applicable terms.

## Loading route A: checkpoint warm start

ConvNeXt checkpoints were trained with a 6-layer / 256-wide decoder — **not** the
variant spec defaults (3/4 layers × 384 width) — so loading without overrides is
rejected by the strict loader. `user_held_model_args(<asset>)` returns all
required `model_args` in one call (including `backbone_pretrained=False` and
`image_size`).

```python
import autovision
from autovision.registry.user_held_assets import user_held_model_args

autovision.train_object_detection(
    out="runs/d1", data="data.yaml", model="detr/dinov3-convnext-base",
    model_args=user_held_model_args("dinov3-convnext-base"),
)
```

All four ConvNeXt tiers require `dec_layers=6` + `decoder_hidden_dim=256`;
`user_held_model_args` already includes them.

### STA-era ViT assets (audit-only, not loadable)

| Asset | Status |
|---|---|
| `dinov3-vitt16` | contains `backbone.sta.*`; loading rejected |
| `dinov3-vitt16plus` | contains `backbone.sta.*`; loading rejected |
| `dinov3-vits16` | contains `backbone.sta.*`; loading rejected |

The DEIMv2-derived spatial prior was removed; the no-STA ViT adapter cannot
faithfully reproduce these checkpoints. Name resolvers (`user_held_model_args`,
`resolve_user_held_asset`, `resolve_avwarm_start_source`,
`resolve_detector_checkpoint`) reject them **before reading/hashing the
container**. For a direct STA-era `.pt`, `load_baseline_coco_checkpoint` /
`load_warm_start_into_network` first read the container to detect
`backbone.sta.*` keys, then reject before building/transferring the model. Error
messages list the four loadable ConvNeXt alternatives.

- Loadable assets: `USER_HELD_LOADABLE_ASSETS` (four ConvNeXt).
- Audit-only assets: `USER_HELD_UNSUPPORTED_ASSETS` (three ViT); the full
  `USER_HELD_ASSETS` map still keeps all seven identity records.
- Audit tooling can read metadata with
  `resolve_user_held_asset(name, allow_unsupported=True)`;
  `verify_user_held_asset(USER_HELD_ASSETS[name])` can still verify sha256
  offline, but **verification does not imply loadability**.
- Migration: retrain/export a no-STA checkpoint, or use a ConvNeXt asset.

## Loading route B: AVWarmStart v1

Our key layout, with shape-compatible partial transfer:

```python
import autovision

autovision.train_object_detection(
    out="runs/d2", data="data.yaml", model="detr/dinov3-convnext-base",
    model_args={
        "warm_start": "dinov3-convnext-base",   # or a warm_start.pt path
        "backbone_pretrained": False,
        "dec_layers": 6,
        "decoder_hidden_dim": 256,
        "image_size": (640, 640),
    },
)
```

## Offline verification

`verify_user_held_asset(asset)` only computes file sha256 and does not load a
model; the asset object must come from the registry (`USER_HELD_ASSETS[name]`, or
a resolver result obtained with `allow_unsupported=True`). A content mismatch
raises. See the [model zoo](model_zoo.md) for the mapping between loading model
keys and build overrides.
