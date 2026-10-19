# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DINOv2 backbone wrapper on top of ``transformers``.

Per framework policy, DINOv2 is provided through HuggingFace
``transformers`` (``Dinov2Model``) instead of vendored code. Pretrained
weights load ``from_pretrained`` (network required); every model also has a
``-notpretrained`` variant built locally from the config (offline).

Wrapping semantics follow the baseline DINOv2 wrapper: the pooled feature is
the CLS token; spatial features strip the CLS + register tokens.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor, nn

from autovision.encoders.model_wrapper import (
    ArchTraits,
    FeatureMapsOut,
    PatchSize,
    PoolOut,
)

__all__ = ["TransformersDinov2Wrapper"]


def _kept_patch_indices(
    mask: Tensor, num_h_patches: int, num_w_patches: int
) -> Tensor:
    """Expand a window-grid keep mask to flat kept-patch indices.

    The mask is a boolean ``(H // window, W // window)`` grid over the same
    canvas as the patch grid ``(num_h_patches, num_w_patches)``. Each window
    covers ``window_size // patch_size`` patches per side, derived here from
    the row counts — which makes any grid that does not evenly tile the
    patch grid (a mask built for a different canvas, or padded to a batch
    grid) a hard error rather than a silent mis-expansion (the reference
    implementation's silent ``F.pad`` crop is exactly the failure mode
    design doc 17 §2.3 calls out).
    """
    grid_h, grid_w = mask.shape
    if grid_h < 1 or grid_w < 1 or num_h_patches % grid_h or num_w_patches % grid_w:
        raise ValueError(
            f"Keep-mask grid {tuple(mask.shape)} does not evenly tile the "
            f"{num_h_patches}x{num_w_patches} patch grid — the mask was not "
            "built for this canvas (design doc 17 §4.2)."
        )
    window_patches = num_h_patches // grid_h
    expanded = mask
    if window_patches > 1:
        expanded = mask.repeat_interleave(window_patches, dim=0).repeat_interleave(
            window_patches, dim=1
        )
    return (
        expanded[:num_h_patches, :num_w_patches]
        .reshape(-1)
        .nonzero(as_tuple=True)[0]
    )


def _backfill_patch_tokens(
    hidden: Tensor,
    kept: Tensor,
    num_prefix: int,
    num_h_patches: int,
    num_w_patches: int,
) -> Tensor:
    """Zero-fill a masked tap back to the dense prefix+patches layout.

    ``hidden`` is a masked-forward tap with the kept patch tokens in kept
    order: the dropped patch positions are re-created as exact zeros via
    ``index_copy_`` (gradients flow through into the kept tokens) and the
    prefix tokens (CLS + registers) are re-attached, so the result matches
    the dense ``(1, prefix + N, D)`` layout the dense path and
    :meth:`_reshape_patch_tokens` expect. Shared by the DINOv2 and DINOv3
    masked forwards (design docs 17/20).
    """
    patch_out = hidden.new_zeros(
        (1, num_h_patches * num_w_patches, hidden.shape[-1])
    )
    patch_out.index_copy_(1, kept, hidden[:, num_prefix:])
    return torch.cat((hidden[:, :num_prefix], patch_out), dim=1)


class TransformersDinov2Wrapper(nn.Module, PatchSize):
    """Adapts a ``transformers.Dinov2Model`` to the ``EncoderProtocol`` protocol."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self._model = model  # transformers models are nn.Modules: register.

    @property
    def hf_model(self) -> nn.Module:
        return self._model

    def raw_module(self) -> nn.Module:
        return self._model

    @property
    def config(self) -> Any:
        return self._model.config

    # ── EncoderProtocol protocol ───────────────────────────────────────────
    def feature_maps(
        self, x: Tensor, bool_masked_pos: Tensor | None = None
    ) -> FeatureMapsOut:
        if bool_masked_pos is None:
            outputs = self._model(pixel_values=x)
        else:
            if not getattr(self.config, "use_mask_token", False):
                raise ValueError(
                    "bool_masked_pos requires the HF mask token, which is "
                    "disabled by default; build the model with "
                    "model_args={'use_mask_token': True}."
                )
            # iBOT-style masking: masked patch positions are replaced by the
            # model's learned mask token (HF Dinov2Model embeddings).
            outputs = self._model(pixel_values=x, bool_masked_pos=bool_masked_pos)
        hidden = outputs.last_hidden_state  # (B, 1 + R + N, D)
        features, cls_token = self._reshape_patch_tokens(hidden, x)
        return {"features": features, "cls_token": cls_token}

    def tap_hidden(self, hidden: Tensor) -> Tensor:
        """Per-tap post-processing of one block hidden state (see
        :meth:`stage_feature_maps` for the per-family semantics).

        The detector baselines tap through ``get_intermediate_layers``;
        the two baseline forks disagree on its ``norm`` flag, so the family
        split lives here:

        - the DINOv2 fork this wrapper ports leaves taps at their RAW
          block output (the historical behavior; ``tests/
          test_models_dinov2.py::test_multiscale_final_tap_pre_layernorm``
          pins it), and
        - the DINOv3 fork applies the model's final LayerNorm to every tap
          (``norm=True`` default) — ``TransformersDinov3ViTWrapper``
          overrides this hook with the measured fix for that family.

        ``feature_maps`` deliberately bypasses this hook: HF's
        ``last_hidden_state`` is already final-normed.
        """
        return hidden

    def stage_feature_maps(
        self, x: Tensor, layer_indices: Sequence[int]
    ) -> list[dict[str, Any]]:
        """Features after the given 0-based block indices (baseline
        ``get_intermediate_layers`` semantics: patch tokens reshaped to
        (B, D, H, W), CLS token per tap, tap post-processing per
        :meth:`tap_hidden`).

        Implemented via a full forward with ``output_hidden_states=True``;
        cost is identical to a plain forward whenever the last tap is the
        final block (true for all registry tap sets).
        """
        layer_indices = list(layer_indices)  # materialize: iterators would be
        # consumed by the validation loop below, silently returning no taps.
        num_blocks = int(self.config.num_hidden_layers)
        for layer_index in layer_indices:
            if not 0 <= layer_index < num_blocks:
                raise ValueError(
                    f"layer_index {layer_index} out of range for a model with "
                    f"{num_blocks} blocks (expected 0..{num_blocks - 1})."
                )
        outputs = self._model(pixel_values=x, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # (embeddings, block_0, ...)
        results: list[dict[str, Any]] = []
        for layer_index in layer_indices:
            features, cls_token = self._reshape_patch_tokens(
                self.tap_hidden(hidden_states[layer_index + 1]), x
            )
            results.append({"features": features, "cls_token": cls_token})
        return results

    def supports_masked_forward(self) -> bool:
        """Whether :meth:`stage_feature_maps_masked` can run.

        The masked path orchestrates the HF model's internals directly
        (``embeddings`` -> ``encoder``), which requires the DINOv2 layout.
        The DINOv3 ViT splits this differently (``embeddings``/``model``/
        ``norm`` plus an explicit rope pass) and implements its own masked
        forward (design doc 20); wrappers that expose neither layout return
        False here, so the caller can surface a config-time error instead
        of a mid-training failure (design docs 17 §4.3/§9, 20 §3.2).
        """
        hf_model = self._model
        return hasattr(hf_model, "embeddings") and hasattr(hf_model, "encoder")

    def stage_feature_maps_masked(
        self,
        x: Tensor,
        layer_indices: Sequence[int],
        keep_masks: Sequence[Tensor | None],
    ) -> list[dict[str, Any]]:
        """Multi-scale taps with per-image background patch-token dropping.

        Same tap semantics as :meth:`stage_feature_maps`, but the
        encoder runs once per image over only its kept patch tokens (design
        doc 17): the keep mask (window grid) is expanded to the patch grid,
        the kept tokens are gathered right after the embeddings stage, and
        every tap's dropped patch positions are zero-filled so the output
        shapes match the dense path exactly — everything downstream (STA
        fusion, pyramid resize, projection) is untouched. Images whose mask
        is ``None`` run the plain dense per-image forward.

        Note: unlike the reference windowed-attention fork, a partial keep
        here CHANGES the attention context of the remaining tokens (the
        stock model is global-attention) — the intended structured-missing
        augmentation, not a lossless speedup (design doc 17 §5).
        """
        if not self.supports_masked_forward():
            raise ValueError(
                "Masked multi-scale forward requires the DINOv2 internals "
                "layout (model.embeddings + model.encoder); this backbone "
                "exposes a different one — design docs 17 §4.3 / 20 §3.2."
            )
        layer_indices = list(layer_indices)
        num_blocks = int(self.config.num_hidden_layers)
        for layer_index in layer_indices:
            if not 0 <= layer_index < num_blocks:
                raise ValueError(
                    f"layer_index {layer_index} out of range for a model with "
                    f"{num_blocks} blocks (expected 0..{num_blocks - 1})."
                )
        if len(keep_masks) != x.shape[0]:
            raise ValueError(
                f"Got {len(keep_masks)} keep masks for {x.shape[0]} images."
            )
        # Same prefix formula as _reshape_patch_tokens (CLS + registers).
        num_prefix = 1 + int(getattr(self.config, "num_register_tokens", 0) or 0)
        patch_size = self.patch_size()
        per_tap: list[list[dict[str, Any]]] = [[] for _ in layer_indices]
        for image_index in range(x.shape[0]):
            image = x[image_index : image_index + 1]
            mask = keep_masks[image_index]
            num_h_patches = image.shape[2] // patch_size
            num_w_patches = image.shape[3] // patch_size
            kept: Tensor | None = None
            if mask is not None:
                kept = _kept_patch_indices(
                    mask.to(device=image.device), num_h_patches, num_w_patches
                )
            embeddings = self._model.embeddings(pixel_values=image)
            if kept is not None:
                # A kept sequence runs through gather/refill even when it
                # covers every patch: index_select of all indices reproduces
                # the dense values exactly, keeping the keep-all ≡ dense
                # invariant honest under test.
                prefix, patches = (
                    embeddings[:, :num_prefix],
                    embeddings[:, num_prefix:],
                )
                embeddings = torch.cat((prefix, patches.index_select(1, kept)), dim=1)
            encoder_outputs = self._model.encoder(
                embeddings, output_hidden_states=True
            )
            hidden_states = encoder_outputs.hidden_states
            for tap_index, layer_index in enumerate(layer_indices):
                hidden = hidden_states[layer_index + 1]
                if kept is not None:
                    # Zero-fill the dropped patch positions so the tap
                    # matches the dense (1, prefix + N, D) layout; gradients
                    # flow through index_copy_ into the kept tokens.
                    hidden = _backfill_patch_tokens(
                        hidden, kept, num_prefix, num_h_patches, num_w_patches
                    )
                # tap_hidden before the reshape, exactly as in the dense
                # path; the tap norm is per-token, so applying it to the
                # backfilled sequence matches the baseline's norm-then-strip
                # order bit-for-bit on the kept tokens.
                features, cls_token = self._reshape_patch_tokens(
                    self.tap_hidden(hidden), image
                )
                per_tap[tap_index].append(
                    {"features": features, "cls_token": cls_token}
                )
        return [
            {
                "features": torch.cat(
                    [image_out["features"] for image_out in tap_images], dim=0
                ),
                "cls_token": torch.cat(
                    [image_out["cls_token"] for image_out in tap_images], dim=0
                ),
            }
            for tap_images in per_tap
        ]

    def pool_features(self, x: FeatureMapsOut) -> PoolOut:
        # Baseline semantics: DINOv2 pools via the CLS token, not avg pool.
        cls_token = x.get("cls_token")
        if cls_token is None:
            raise ValueError("pool_features requires 'cls_token' from feature_maps.")
        return {"pooled_features": cls_token[..., None, None]}

    def encoder_dim(self) -> int:
        return int(self.config.hidden_size)

    def patch_size(self) -> int:
        return int(self.config.patch_size)

    def arch_traits(self) -> ArchTraits:
        return {"model_type": "transformer", "norm_type": "layernorm"}

    def make_teacher(self) -> "TransformersDinov2Wrapper":
        """Prepare for teacher duty: eval mode and no stochastic dropout.

        Also zeroes DropPath-style stochastic depth: HF ``Dinov2DropPath``
        subclasses ``nn.Module`` (attribute ``drop_prob``), not ``nn.Dropout``,
        so ``eval()`` alone would only keep it dormant and a later ``train()``
        call would silently re-enable it.
        """
        self.eval()
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if hasattr(module, "drop_prob"):
                module.drop_prob = 0.0
        return self

    # ── Internal helpers ────────────────────────────────────────────────
    def _patch_grid(self, image: Tensor) -> tuple[int, int]:
        """Patch-grid dims from the INPUT size (``H // patch``, ``W // patch``).

        Derived from the input rather than recovered from the token count:
        under dynamic H/W (roadmap D2) the token count is a symbolic value
        and an ``isqrt`` reconstruction cannot be traced.
        """
        patch_size = self.patch_size()
        return image.shape[-2] // patch_size, image.shape[-1] // patch_size

    def _reshape_patch_tokens(self, hidden: Tensor, image: Tensor) -> tuple[Tensor, Tensor]:
        """Split the prefix tokens off a hidden state and reshape patches.

        Strips the CLS + register tokens and reshapes the patch tokens to a
        square (B, C, H, W) map; the grid comes from ``image``'s spatial size
        (dynamic-shape safe). Returns ``(features, cls_token)``.
        """
        cls_token = hidden[:, 0]
        num_prefix = 1 + int(getattr(self.config, "num_register_tokens", 0) or 0)
        patches = hidden[:, num_prefix:]  # (B, N, D)
        channels = self.encoder_dim()
        grid_h, grid_w = self._patch_grid(image)
        if not torch.compiler.is_compiling():
            # Eager safety net, kept OUT of traced graphs: with dynamic H/W
            # the grid and the token count are symbolic and data-dependent
            # comparisons cannot be traced (the reshape below is the runtime
            # guard there).
            if grid_h != grid_w:
                raise ValueError(
                    f"Non-square DINOv2 feature map ({grid_h}x{grid_w} patches) "
                    "is not supported."
                )
            if patches.shape[1] != grid_h * grid_w:
                raise ValueError(
                    f"Patch grid {grid_h}x{grid_w} does not match "
                    f"{patches.shape[1]} patch tokens."
                )
        features = (
            patches.transpose(1, 2)
            .reshape(patches.shape[0], channels, grid_h, grid_w)
            .contiguous()
        )
        return features, cls_token
