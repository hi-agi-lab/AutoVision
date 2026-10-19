# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR package, backed by HuggingFace ``transformers``.

Per framework policy (see the DINOv2/DINOv3 packages), RF-DETR is provided
through ``transformers`` (``RfDetrModel``) instead of the Roboflow
``rfdetr`` package: HF models are plain ``nn.Module``\ s and the DINOv2
backbone exposes multi-scale feature maps directly. transformers gained
RF-DETR support in 5.9.0 (2026-05), so this package is **gated**: it
auto-enables when the installed transformers exposes the RfDetr classes
and degrades gracefully (empty model list, informative error) otherwise.

Checkpoints are the official Roboflow conversions on the Hub
(``Roboflow/rf-detr-*``). Each model also has a ``-notpretrained``
offline variant built locally from the config. The ``seg-preview`` and
other instance-segmentation variants are not listed: they load into
``RfDetrForInstanceSegmentation``, not the bare ``RfDetrModel`` this
package wraps.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import torch

from autovision.encoders.package import CatalogModelSource
from autovision.encoders.rfdetr.rfdetr import RFDETRModelWrapper
from autovision.encoders.weight_verification import load_pretrained_verified

__all__ = ["RFDETR_SOURCE", "RFDETRSource"]

# All RF-DETR variants share the distilled DINOv2 backbone architecture;
# variants differ in patch/window layout, input size and decoder depth
# (per the Roboflow/rf-detr-* checkpoint configs). The config derives
# window_block_indexes from out_indices, so only the layout the config
# cannot infer is tabulated here.
_HIDDEN_SIZE = 384
_NUM_HIDDEN_LAYERS = 12
_NUM_ATTENTION_HEADS = 6
_IMAGE_SIZE = 224  # local untrained builds; divisible by patch*windows for every variant

# model name -> (HF checkpoint, decoder depth, backbone layout)
_MODEL_INFO: dict[str, dict[str, Any]] = {
    "rf-detr-base": dict(
        hf="Roboflow/rf-detr-base",
        decoder_layers=3,
        backbone=dict(
            patch_size=14,
            num_windows=4,
            out_indices=[2, 5, 8, 11],
            out_features=["stage2", "stage5", "stage8", "stage11"],
        ),
    ),
    "rf-detr-nano": dict(
        hf="Roboflow/rf-detr-nano",
        decoder_layers=2,
        backbone=dict(
            patch_size=16,
            num_windows=2,
            out_indices=[3, 6, 9, 12],
            out_features=["stage3", "stage6", "stage9", "stage12"],
        ),
    ),
    "rf-detr-small": dict(
        hf="Roboflow/rf-detr-small",
        decoder_layers=3,
        backbone=dict(
            patch_size=16,
            num_windows=2,
            out_indices=[3, 6, 9, 12],
            out_features=["stage3", "stage6", "stage9", "stage12"],
        ),
    ),
    "rf-detr-medium": dict(
        hf="Roboflow/rf-detr-medium",
        decoder_layers=4,
        backbone=dict(
            patch_size=16,
            num_windows=2,
            out_indices=[3, 6, 9, 12],
            out_features=["stage3", "stage6", "stage9", "stage12"],
        ),
    ),
    "rf-detr-large": dict(
        hf="Roboflow/rf-detr-large",
        decoder_layers=4,
        backbone=dict(
            patch_size=16,
            num_windows=2,
            out_indices=[3, 6, 9, 12],
            out_features=["stage3", "stage6", "stage9", "stage12"],
        ),
    ),
}


def _rfdetr_classes() -> dict[str, Any] | None:
    """Live detection of RF-DETR support in the installed transformers.

    Returns the class map or ``None`` when transformers is missing or older
    than 5.9. Reads module attributes (not imports) so the check stays
    honest across transformers versions with differing class spellings.
    """
    try:
        transformers = importlib.import_module("transformers")
    except ImportError:
        return None
    config = getattr(transformers, "RfDetrConfig", None) or getattr(
        transformers, "RFDetrConfig", None
    )
    model = getattr(transformers, "RfDetrModel", None) or getattr(
        transformers, "RFDetrModel", None
    )
    backbone_config = getattr(transformers, "RfDetrDinov2Config", None) or getattr(
        transformers, "RFDetrDinov2Config", None
    )
    if None in (config, model, backbone_config):
        return None
    return {"config": config, "model": model, "backbone_config": backbone_config}


def _support_error(model_name: str) -> ImportError:
    return ImportError(
        f"RF-DETR model '{model_name}' requires a transformers version with "
        f"RF-DETR support (classes RfDetrModel/RfDetrConfig, transformers "
        f">= 5.9). Upgrade: pip install -U 'autovision[rfdetr]'."
    )


class RFDETRSource(CatalogModelSource):
    name = "rfdetr"

    @classmethod
    def catalog_names(cls) -> list[str]:
        if _rfdetr_classes() is None:
            return []
        names: list[str] = []
        for model_name in _MODEL_INFO:
            names.append(f"{cls.name}/{model_name}")
            names.append(f"{cls.name}/{model_name}-notpretrained")
        return sorted(names)

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        if isinstance(model, RFDETRModelWrapper):
            return True
        classes = _rfdetr_classes()
        if classes is None:
            return False
        return isinstance(model, classes["model"])

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> Any:
        if num_input_channels != 3:
            raise ValueError(
                f"RF-DETR models only support 3 input channels, got "
                f"{num_input_channels}."
            )
        classes = _rfdetr_classes()
        if classes is None:
            raise _support_error(model_name)
        base_name = model_name.removesuffix("-notpretrained")
        pretrained = model_name == base_name
        if base_name not in _MODEL_INFO:
            supported = ", ".join(sorted(_MODEL_INFO))
            raise ValueError(
                f"Unknown rfdetr model: '{model_name}'. Supported models: "
                f"{supported}."
            )
        info = _MODEL_INFO[base_name]
        config_kwargs = dict(model_args or {})
        if pretrained and load_weights:
            model = load_pretrained_verified(classes["model"], info["hf"], config_kwargs)
        else:
            # Local (offline) build from the config. The class defaults for
            # RfDetrConfig already match the base variant; per-variant
            # overrides below mirror the checkpoint configs.
            backbone_kwargs = dict(info["backbone"])
            backbone_kwargs["image_size"] = _IMAGE_SIZE
            backbone_config = classes["backbone_config"](
                hidden_size=_HIDDEN_SIZE,
                num_hidden_layers=_NUM_HIDDEN_LAYERS,
                num_attention_heads=_NUM_ATTENTION_HEADS,
                use_mask_token=True,
                **backbone_kwargs,
            )
            config_kwargs.setdefault("decoder_layers", info["decoder_layers"])
            config = classes["config"](
                backbone_config=backbone_config, **config_kwargs
            )
            model = classes["model"](config)
        return model

    @classmethod
    def as_encoder(cls, model: Any) -> RFDETRModelWrapper:
        if isinstance(model, RFDETRModelWrapper):
            return model
        return RFDETRModelWrapper(model)

    @classmethod
    def export_weights(cls, model: Any, out: Path, log_example: bool = True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        wrapper = cls.as_encoder(model)
        torch.save(wrapper.raw_module().state_dict(), out)
        if log_example:
            print(
                f"Exported RF-DETR model to '{out}'. Rebuild with "
                f"autovision.get_wrapped_model('<rfdetr model name>') and "
                f"load_state_dict(torch.load('{out}'))."
            )


RFDETR_SOURCE = RFDETRSource()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(RFDETR_SOURCE)
