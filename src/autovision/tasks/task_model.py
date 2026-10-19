# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""TaskNetwork: pure-PyTorch, three-stage inference model (exportable).

Task models are the deployment-facing artifacts: they must not import any
training-only machinery. ``load_model`` rebuilds them from
``(class_path, init_args, state_dict)`` in exported files.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from autovision.encoders._model_helpers import load_state_dict_compat

__all__ = ["TaskNetwork"]


class TaskNetwork(nn.Module):
    """Base class for all task models (pure PyTorch inference)."""

    def __init__(
        self,
        init_args: dict[str, Any],
        ignore_args: set[str] | None = None,
    ) -> None:
        super().__init__()
        # ``init_args=locals()`` captures the ``__class__`` cell too. It is
        # not a constructor argument, and persisting a class object in an
        # export makes the pickle-free loader reject an otherwise safe file.
        ignore = set(ignore_args or ()) | {"__class__"}
        self._init_args: dict[str, Any] = {
            key: value for key, value in init_args.items() if key not in ignore
        }

    @property
    def init_args(self) -> dict[str, Any]:
        return dict(self._init_args)

    @property
    def class_path(self) -> str:
        return f"{type(self).__module__}.{type(self).__name__}"

    def deploy(self) -> "TaskNetwork":
        """Switch to deployment mode (recursive convert_to_deploy hooks)."""
        for module in self.modules():
            convert = getattr(module, "convert_to_deploy", None)
            if callable(convert):
                convert()
        return self

    # ── three-stage inference contract ─────────────────────────────────
    def preprocess_image(
        self, image: Any
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        return batch

    def forward_backend(self, x: torch.Tensor) -> Any:
        raise NotImplementedError

    def postprocess(self, raw_outputs: Any, metadata: list[dict[str, Any]], **kwargs) -> list[Any]:
        raise NotImplementedError

    def _module_device(self) -> torch.device:
        """Device the network's parameters (or buffers) live on."""
        parameter = next(self.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(self.buffers(), None)
        return buffer.device if buffer is not None else torch.device("cpu")

    @torch.no_grad()
    def predict(self, image: Any, **kwargs) -> Any:
        """Single-image convenience: preprocess -> forward -> postprocess.

        Loading/normalization stays on CPU; the tensor is moved onto the
        network's device before the forward, so a network moved with
        ``.to("cuda")`` serves path/PIL inputs directly.
        """
        was_training = self.training
        self.eval()
        tensor, meta = self.preprocess_image(image)
        batch = self.preprocess_batch(tensor.unsqueeze(0))
        raw = self.forward_backend(batch.to(self._module_device()))
        results = self.postprocess(raw, [meta], **kwargs)
        if was_training:
            self.train()
        return results[0]

    def _load_image_unchanged(self, image: Any) -> torch.Tensor:
        """path/PIL/tensor -> (C, H, W) float tensor in [0, 1], NO resize.

        The loading/normalization contract mirrors the task models'
        ``preprocess_image`` readers; used by :meth:`predict_sahi` which
        needs the original-resolution canvas for slicing.
        """
        import pathlib

        from torchvision.transforms.v2 import functional as TF

        from autovision.data.file_helpers import open_image

        if isinstance(image, (str, pathlib.Path)):
            tensor = open_image(image).float() / 255.0
        elif isinstance(image, torch.Tensor):
            tensor = image.float()
            if tensor.ndim == 3 and tensor.max() > 1.5:
                tensor = tensor / 255.0
        else:
            tensor = TF.pil_to_tensor(image).float() / 255.0
        return tensor

    @torch.no_grad()
    def predict_sahi(
        self,
        image: Any,
        tile_size: int | None = None,
        overlap: float = 0.2,
        threshold: float = 0.6,
        nms_iou_threshold: float = 0.3,
        global_local_iou_threshold: float = 0.1,
        padding_mode: str = "resize",
        batch_size: int | None = None,
        merge: str = "nms",
        overlap_metric: str = "iou",
    ) -> dict[str, torch.Tensor]:
        """Slicing-aided hyper inference (SAHI) for small objects.

        The original-resolution image is tiled with overlap (baseline
        semantics: tile_size defaults to the model input size, a resized
        full-image view serves as the global tile), per-view predictions are
        filtered by ``threshold``, then merged by
        ``combine_object_detection_tiles`` (per-label NMS/NMM, tile boxes
        offset to original coordinates, tiles duplicating a same-label
        global box suppressed or merged).

        ``batch_size`` bounds how many views (global + tiles) go through one
        forward pass; ``None`` keeps the historical single-batch behavior.
        On large rasters this prevents the tile stack from multiplying peak
        VRAM. ``merge="nmm"`` unions overlapping fragments (better for
        truncated/adjacent objects); ``overlap_metric="ios"`` makes the
        overlap test sensitive to nested boxes.

        Only object-detection task models (postprocess output with
        ``bboxes``/``scores``/``labels`` tensors) are supported; anything
        else raises :class:`TypeError`.

        Tile coordinate bookkeeping: each tile is model-preprocessed to
        ``image_size`` while its metadata records the *effective* canvas
        (the un-padded edge size for ``pad`` mode, the full tile for
        ``resize`` mode), so decoded boxes land back in the tile's original
        frame before the offset to the original image.
        """
        from autovision.tasks.shared.tiling_utils import (
            combine_object_detection_tiles,
            tile_image,
        )

        if batch_size is not None and batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 or None, got {batch_size}.")
        was_training = self.training
        self.eval()
        try:
            original = self._load_image_unchanged(image)
            _, height, width = original.shape
            if tile_size is None:
                tile_size = int(min(self.image_size))
            tiles, coordinates = tile_image(
                original, overlap, tile_size, padding_mode=padding_mode
            )

            # Effective canvas per tile: resize-mode edge tiles are
            # interpolated to the full tile, so their boxes are decoded in
            # the *pre-interpolation* edge frame; pad-mode tiles keep the
            # full tile frame (out-of-canvas detections are clipped during
            # merging).
            tile_metas = []
            for x_start, y_start in coordinates.tolist():
                if padding_mode == "resize":
                    edge_h = min(tile_size, height - y_start)
                    edge_w = min(tile_size, width - x_start)
                else:
                    edge_h = edge_w = tile_size
                tile_metas.append({"orig_w": edge_w, "orig_h": edge_h})

            # Global view (resized full image) + all tiles; optional chunking
            # bounds peak memory on large rasters.
            global_tensor, global_meta = self.preprocess_image(original)
            tile_tensors = []
            for tile in tiles:
                tile_tensor, _ = self.preprocess_image(tile)
                tile_tensors.append(tile_tensor)
            views = [global_tensor, *tile_tensors]
            metas = [global_meta, *tile_metas]
            chunk = batch_size or len(views)
            results: list[Any] = []
            for start in range(0, len(views), chunk):
                batch = torch.stack(views[start : start + chunk])
                results.extend(
                    self.postprocess(
                        self.forward_backend(
                            self.preprocess_batch(batch).to(self._module_device())
                        ),
                        metas[start : start + chunk],
                        threshold=threshold,
                    )
                )

            global_output = results[0]
            if "bboxes" not in global_output:
                raise TypeError(
                    f"predict_sahi supports object-detection task models "
                    f"whose postprocess yields 'bboxes'/'scores'/'labels'; "
                    f"{type(self).__name__} produced "
                    f"{sorted(global_output)}."
                )
            return combine_object_detection_tiles(
                global_output,
                results[1:],
                coordinates,
                (width, height),
                nms_iou_threshold=nms_iou_threshold,
                global_local_iou_threshold=global_local_iou_threshold,
                merge=merge,
                overlap_metric=overlap_metric,
            )
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def predict_batch(self, images: list[Any], **kwargs) -> list[Any]:
        self.eval()
        tensors, metadata = [], []
        for image in images:
            tensor, meta = self.preprocess_image(image)
            tensors.append(tensor)
            metadata.append(meta)
        batch = self.preprocess_batch(torch.stack(tensors))
        return self.postprocess(
            self.forward_backend(batch.to(self._module_device())), metadata, **kwargs
        )

    # ── checkpoint loading ──────────────────────────────────────────────
    def load_train_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load a training checkpoint into this model.

        Training checkpoints store the EMA-prefixed weights when EMA is
        enabled (preferred) or the plain ``model.`` prefix; the prefix is
        stripped before loading.
        """
        prefixes = ("ema_model.model.", "ema_model.", "model.")
        for prefix in prefixes:
            stripped = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if stripped:
                load_state_dict_compat(self, stripped)
                return
        load_state_dict_compat(self, state_dict)
