# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Compatibility shims for known third-party defects.

Imported once by :mod:`autovision` (see package ``__init__``) so that every
``copy.deepcopy`` site in the codebase — EMA trackers, distillation teachers,
export benchmarks — is protected from the GraphModule defect below.
"""

from __future__ import annotations

import copy

__all__ = ["install_graphmodule_deepcopy_fix"]

_PATCH_FLAG = "_autovision_graphmodule_deepcopy_fix"
# Plain attributes torchvision's GraphModule stores in __init__ and swaps in
# train(); torch.fx.GraphModule.__deepcopy__ does not carry them over.
_GRAPH_ATTRS = ("train_graph", "eval_graph")


def install_graphmodule_deepcopy_fix() -> None:
    """Keep torchvision ``GraphModule`` train/eval graphs alive across deepcopy.

    ``torch.fx.GraphModule.__deepcopy__`` rebuilds the copy with ``__new__``
    plus the fx base-class ``__init__`` and only restores a fixed list of
    hook attributes. torchvision's ``feature_extraction.GraphModule`` and
    ``DualGraphModule`` (used by the RT-DETR / DFINE hybrid encoders) store
    ``train_graph`` / ``eval_graph`` as plain attributes and swap
    ``self.graph`` inside ``train()`` — the fx copy drops them, so any
    ``.eval()`` / ``.train()`` on a deepcopied model raises
    ``AttributeError: 'GraphModule' object has no attribute 'eval_graph'``.

    The fx implementation already deep-copied the full ``__dict__`` into an
    intermediate module, so re-copying the two graphs here is served by the
    shared memo. Idempotent; a no-op without torchvision.
    """
    try:
        from torchvision.models import feature_extraction as _fe
    except ImportError:  # torchvision is optional for pure-pretraining flows
        return
    for graph_module_cls in (
        getattr(_fe, "GraphModule", None),
        getattr(_fe, "DualGraphModule", None),
    ):
        if graph_module_cls is None or getattr(graph_module_cls, _PATCH_FLAG, False):
            continue
        setattr(graph_module_cls, _PATCH_FLAG, True)
        fx_deepcopy = graph_module_cls.__deepcopy__

        def __deepcopy__(self, memo, _fx_deepcopy=fx_deepcopy):  # noqa: N807
            res = _fx_deepcopy(self, memo)
            for attr in _GRAPH_ATTRS:
                if attr in self.__dict__ and not hasattr(res, attr):
                    setattr(res, attr, copy.deepcopy(self.__dict__[attr], memo))
            return res

        graph_module_cls.__deepcopy__ = __deepcopy__
