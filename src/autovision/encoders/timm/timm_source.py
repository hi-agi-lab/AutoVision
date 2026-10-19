# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""timm model catalog as a :class:`CatalogModelSource`.

Every model registered in timm is buildable under a ``timm/<name>`` catalog
name, with any number of input channels. Models are built **untrained**
unless the caller explicitly opts into weights through ``model_args``
(``{"pretrained": True}`` or a ``checkpoint_path``); downloading pretrained
weights additionally needs network access. timm itself is an optional extra:
without it the catalog is simply empty instead of erroring.
"""

from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from autovision.encoders.package import CatalogModelSource
from autovision.encoders.timm.timm_wrapper import (
    TIMMModelWrapper,
    pretrained_architecture,
)

__all__ = ["TIMM_SOURCE", "TimmSource"]


def _load_timm() -> ModuleType | None:
    """The timm module, imported lazily; ``None`` when the extra is absent."""
    try:
        return import_module("timm")
    except ImportError:
        return None


def _load_timm_or_complain(model_name: str) -> ModuleType:
    """The timm module, or a ``ValueError`` explaining how to install it."""
    try:
        return import_module("timm")
    except ImportError as error:
        raise ValueError(
            f"Cannot create model 'timm/{model_name}' because timm is not "
            "installed. Install it with: pip install 'autovision[timm]'."
        ) from error


def _underlying(model: Any) -> Any:
    """Unwrap an adapter to the raw module it was built around."""
    if isinstance(model, TIMMModelWrapper):
        return model.raw_module()
    return model


def _has_timm_ancestor(model: Any) -> bool:
    """Whether any class of the module's lineage lives in the timm package."""
    for ancestor in type(model).__mro__:
        if ancestor.__module__.partition(".")[0] == "timm":
            return True
    return False


# Families whose patch embedding rejects image sizes other than the trained
# one unless variable-resolution input is switched on explicitly.
_VARIABLE_RESOLUTION_FAMILIES: tuple[str, ...] = tuple("vit eva deit".split())


class TimmSource(CatalogModelSource):
    """Catalog source for the timm model zoo."""

    name: str = "timm"

    @classmethod
    def recognizes(cls, model: Any) -> bool:
        """Whether ``model`` is a timm adapter or a timm-registered module."""
        return isinstance(model, TIMMModelWrapper) or _has_timm_ancestor(model)

    @classmethod
    def export_weights(cls, model, out: Path, log_example=True) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        backbone = _underlying(model)
        supported = cls.recognizes(backbone)
        if not supported:
            raise ValueError(
                f"The timm package only exports timm models, got "
                f"'{type(backbone).__name__}'."
            )
        torch.save(obj=backbone.state_dict(), f=out)
        if log_example:
            print(
                f"Exported timm model to '{out}'. Load with: "
                f"timm.create_model(model_name="
                f"'{pretrained_architecture(backbone)}', "
                f"checkpoint_path='{out}')."
            )

    @classmethod
    def catalog_names(cls) -> list[str]:
        timm_module = _load_timm()
        if timm_module is None:
            return []
        prefix = cls.name + "/"
        entries = timm_module.list_models()
        return [prefix + entry for entry in entries]

    @classmethod
    def build(
        cls,
        model_name: str,
        num_input_channels: int = 3,
        model_args: dict[str, Any] | None = None,
        load_weights: bool = True,
    ) -> nn.Module:
        timm_module = _load_timm_or_complain(model_name)
        options = {"pretrained": False, "in_chans": num_input_channels}
        if model_name.startswith(_VARIABLE_RESOLUTION_FAMILIES):
            options["dynamic_img_size"] = True
        if model_args:
            options |= model_args
        if not load_weights:
            # Weights were explicitly declined: override any opt-in above.
            options = {**options, "pretrained": False, "checkpoint_path": None}
        return timm_module.create_model(**options, model_name=model_name)

    @classmethod
    def as_encoder(cls, model: nn.Module) -> TIMMModelWrapper:
        return model if isinstance(model, TIMMModelWrapper) else TIMMModelWrapper(model)


TIMM_SOURCE = TimmSource()

from autovision.encoders.package_helpers import register_source  # noqa: E402

register_source(TIMM_SOURCE)
