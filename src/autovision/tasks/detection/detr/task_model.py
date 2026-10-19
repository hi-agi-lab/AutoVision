# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""DETR object detection task model (inference/export facing).

Assembly (baseline SSL-backbone detector): SSL backbone wrapper ->
stride 8/16/32 pyramid -> shared HybridEncoder -> RT-DETRv2 or D-FINE
decoder.
Reuses the framework's shared detection components and the shared top-k
decoder, so the coordinate contract is identical to the RT-DETR task
family. Per-variant encoder/decoder recipes come from the registry spec.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from autovision.augment.transform_args import NormalizeArgs
from autovision.encoders import get_wrapped_model
from autovision.encoders.dinov2.dinov2_wrapper import TransformersDinov2Wrapper
from autovision.errors import CheckpointError
from autovision.export import ONNXExportMixin, ONNXExportPrecisionPolicy
from autovision.tasks.detection.rtdetr.task_model import (
    ObjectDetectionOutput,
)
from autovision.tasks.detection.shared.decode import (
    decode_detection_outputs,
)
from autovision.tasks.detection.shared.dfine_decoder import (
    DFINEv2Decoder,
)
from autovision.tasks.detection.shared.dfine_official_decoder import (
    DFINEOfficialDecoder,
)
from autovision.tasks.detection.shared.hybrid_encoder import (
    HybridEncoder,
)
from autovision.tasks.detection.shared.open_vocab import (
    encode_class_names,
    resolve_text_embed_dim,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    RTDETRv2Decoder,
)
from autovision.tasks.detection.shared.ssl_contrastive import (
    SSLProjectionHead,
)
from autovision.tasks.task_model import TaskNetwork

from .backbone_wrappers import (
    ConvNeXtDetectionBackbone,
    GTRViTBackbone,
    TimmCNNBackbone,
    ViTDetectionBackbone,
)
from .config import (
    DETRDensityArgs,
    DETROpenVocabArgs,
    DETRSSLArgs,
    DETRTokenDropArgs,
    DETRVariantSpec,
    check_image_size,
    resolve_detr_variant,
)
from .dome_decoder import DomeDFINEv2Decoder
from .dome_encoder import DomeHybridEncoder

__all__ = ["DETRObjectDetection"]


def _build_vit_test_backbone() -> TransformersDinov2Wrapper:
    """Tiny offline DINOv2 build for the ``vittest`` registry variant."""
    import transformers

    config = transformers.Dinov2Config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        image_size=224,
        patch_size=14,
        num_register_tokens=0,
    )
    return TransformersDinov2Wrapper(transformers.Dinov2Model(config))


def _relabel_prefix_drift(state: dict, target: torch.nn.Module) -> dict:
    """Absorb cross-version backbone state-dict key renames.

    External ``backbone_weights`` files are converted against one
    transformers layout and consumed under another (5.19 moved the
    DINOv3-ConvNeXt weights under a ``model.`` prefix while the final
    ``layer_norm`` stayed top-level). Applied ONLY when it explains every
    file key — a pure key-space rename; tensors are never touched, and
    any unexplainable key returns the state unchanged so the caller's
    strict check reports the raw names.
    """
    expected = target.state_dict().keys()
    expected_set = set(expected)
    if set(state) == expected_set:
        return state
    relabeled = {}
    for key, value in state.items():
        if key in expected_set:
            relabeled[key] = value
        elif f"model.{key}" in expected_set:
            relabeled[f"model.{key}"] = value
        else:
            return state
    return relabeled


class DETRObjectDetection(ONNXExportMixin, TaskNetwork):
    """SSL-backbone DETR detector (three-stage inference contract)."""

    onnx_export_precision_policy = ONNXExportPrecisionPolicy(
        fp32_onnx_op_types=("Softmax", "MatMul")
    )
    #: Export-side postprocess contract marker (see export/postprocess.py).
    postprocess_spec_family = "detr"

    def __init__(
        self,
        num_classes: int,
        model_name: str = "dinov2-vits14-notpretrained",
        image_size: tuple[int, int] | None = None,
        decoder: Literal["rtdetrv2", "dfine", "dfine_official", "auto"] = "auto",
        reg_max: int = 32,
        reg_scale: float = 4.0,
        enc_dim_feedforward: int | Literal["auto"] = "auto",
        enc_expansion: float | Literal["auto"] = "auto",
        enc_depth_mult: float | Literal["auto"] = "auto",
        hidden_dim: int | None = None,
        decoder_hidden_dim: int | None = None,
        num_queries: int | None = None,
        dec_layers: int | None = None,
        num_heads: int | None = None,
        num_top_queries: int = 300,
        num_denoising: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        dn_max_gt_cap: int | None = None,
        mask_downsample_ratio: int | None = None,
        backbone_pretrained: bool = False,
        backbone_weights: str | None = None,
        load_weights: bool = True,
        pyramid_levels: int = 3,
        density: DETRDensityArgs | dict | None = None,
        ssl: DETRSSLArgs | dict | None = None,
        token_drop: DETRTokenDropArgs | dict | None = None,
        open_vocab: DETROpenVocabArgs | dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            init_args=locals(), ignore_args={"load_weights", "kwargs", "self"}
        )
        # Exported checkpoints are deserialized with the pickle-free safe
        # loader (weights_only=True), which rejects arbitrary objects —
        # snapshot the density/ssl/token-drop configs as plain dicts;
        # __init__ re-coerces dicts back into the args classes below, so
        # the round trip is lossless. Same for Path-valued args: the export
        # safe loader (weights_only=True) rejects pathlib types, and this
        # task model treats backbone_weights as a str either way.
        if isinstance(self._init_args.get("density"), DETRDensityArgs):
            self._init_args["density"] = self._init_args["density"].model_dump()
        if isinstance(self._init_args.get("ssl"), DETRSSLArgs):
            self._init_args["ssl"] = self._init_args["ssl"].model_dump()
        if isinstance(self._init_args.get("token_drop"), DETRTokenDropArgs):
            self._init_args["token_drop"] = self._init_args["token_drop"].model_dump()
        if isinstance(self._init_args.get("open_vocab"), DETROpenVocabArgs):
            self._init_args["open_vocab"] = self._init_args["open_vocab"].model_dump()
        if isinstance(self._init_args.get("backbone_weights"), pathlib.PurePath):
            self._init_args["backbone_weights"] = str(
                self._init_args["backbone_weights"]
            )
        self.num_classes = num_classes
        self.num_top_queries = num_top_queries
        self.normalize = NormalizeArgs()

        # resolve_detr_variant is idempotent: accepts both the public
        # "detr/<backbone>" name and the bare backbone key stored in
        # training init-args / exported checkpoints.
        _, spec = resolve_detr_variant(model_name)
        self.variant_spec: DETRVariantSpec = spec
        if image_size is None:
            # Default 640 snapped UP to the patch grid (644 for patch 14),
            # mirroring the train-args resolution.
            divisor = 2 * spec.patch_size
            side = -(-640 // divisor) * divisor
            self.image_size = (side, side)
        else:
            check_image_size(tuple(image_size), spec)
            self.image_size = tuple(image_size)
        hidden_dim = hidden_dim or spec.hidden_dim
        # Decoder width, decoupled from the encoder width. The baseline
        # runs the decoder at a FIXED 256 while the HybridEncoder width
        # varies per variant (ConvNeXt: 384); the fork historically shared
        # one width. ``None`` keeps the shared width (checkpoint-compatible
        # with every existing fork model); an explicit value enables
        # loading baseline detectors (encoder 384 + decoder 256).
        decoder_hidden_dim = decoder_hidden_dim or hidden_dim
        dec_layers = dec_layers or spec.dec_layers
        num_heads = num_heads or spec.dec_num_heads
        num_queries = num_queries or spec.num_queries
        self.hidden_dim = hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        if decoder == "auto":
            decoder = spec.decoder_default
        if decoder not in ("rtdetrv2", "dfine", "dfine_official"):
            raise ValueError(f"Unknown DETR decoder: '{decoder}'.")
        self.decoder_name = decoder

        # Density-guided branch (Dome-DETR port). ``None`` keeps the
        # baseline contract bit-for-bit: no dome modules are constructed.
        if pyramid_levels not in (3, 4):
            raise ValueError(f"pyramid_levels must be 3 or 4, got {pyramid_levels}.")
        if pyramid_levels == 4 and spec.family != "convnext":
            raise ValueError(
                f"pyramid_levels=4 needs a ConvNeXt-family backbone (the "
                f"only wrapper exposing a stride-4 stage); "
                f"'{spec.backbone_name}' is family '{spec.family}'."
            )
        if isinstance(density, dict):
            density = DETRDensityArgs(**density)
        if density is not None:
            if decoder != "dfine":
                raise ValueError(
                    "density (dynamic queries / DeFE / MWAS) requires the "
                    "D-FINE decoder; pass decoder='dfine' (the evaluated "
                    "v1 combination, design doc 13 §3.3)."
                )
            if num_queries > density.max_num_select:
                raise ValueError(
                    f"num_queries ({num_queries}) must not exceed "
                    f"density.max_num_select ({density.max_num_select})."
                )
        self.density_args: DETRDensityArgs | None = density
        self._density_enabled = density is not None

        # Per-object contrastive SSL branch (design doc 14). ``None`` or
        # ``mode="off"`` keeps the baseline contract: no head is constructed
        # (mode="off" + two_crop is the batch-matched no-SSL baseline — the
        # doubled data pipeline runs, the loss does not).
        if isinstance(ssl, dict):
            ssl = DETRSSLArgs(**ssl)
        if ssl is not None and decoder != "dfine":
            # v1 boundary (design doc 14 §3.3): the RT-DETRv2 decoder has
            # no query_features emission, so the whole ssl config —
            # including the mode="off" batch-matched baseline — is
            # restricted to the D-FINE decoder family.
            raise ValueError(
                "ssl (per-object contrastive head) requires the D-FINE "
                "decoder in v1; pass decoder='dfine' (design doc 14 §3.3)."
            )
        self.ssl_args: DETRSSLArgs | None = ssl
        # Route B (design doc 14 §4.10): label-free SSL-only pretraining off
        # the decoder's top-k candidate features. The density-gated selection
        # is not supported as an instance source (zero-padded candidates).
        self._ssl_candidates_mode = (
            ssl is not None and ssl.instance_source == "encoder_candidates"
        )
        if self._ssl_candidates_mode and self._density_enabled:
            raise ValueError(
                "ssl instance_source='encoder_candidates' (SSL-only "
                "pretraining) is not supported with the density branch; "
                "disable density (design doc 14 §4.10)."
            )

        if spec.requires_dinov3 and not _dinov3_supported():
            raise ValueError(
                f"DETR variant '{model_name}' needs the DINOv3 classes "
                "from 'transformers', which are not available in the "
                "installed version."
            )

        # In-backbone background token dropping (design doc 17). ``None``
        # keeps the baseline contract bit-for-bit; when enabled the
        # training step derives per-sample window keep masks on the final
        # canvas and the ViT backbone skips the dropped tokens.
        if isinstance(token_drop, dict):
            token_drop = DETRTokenDropArgs(**token_drop)
        if token_drop is not None:
            if spec.family != "vit":
                raise ValueError(
                    "token_drop requires a ViT-family backbone (the only "
                    f"one with a patch-token sequence); "
                    f"'{spec.backbone_name}' is family '{spec.family}'."
                )
            if token_drop.window_size % spec.patch_size != 0:
                raise ValueError(
                    f"token_drop.window_size ({token_drop.window_size}) must "
                    f"be a multiple of the backbone patch size "
                    f"({spec.patch_size})."
                )
        self._token_drop_args: DETRTokenDropArgs | None = token_drop

        # Text-contrastive open-vocabulary branch (design doc 27). ``None``
        # keeps the baseline contract bit-for-bit: no text asset is built
        # and the per-class Linear score heads stay in place.
        if isinstance(open_vocab, dict):
            open_vocab = DETROpenVocabArgs(**open_vocab)
        if open_vocab is not None and len(open_vocab.class_names) != num_classes:
            raise ValueError(
                f"open_vocab carries {len(open_vocab.class_names)} class "
                f"names but the model builds num_classes={num_classes}; the "
                f"vocabulary order defines the label space."
            )
        self.open_vocab_args: DETROpenVocabArgs | None = open_vocab

        backbone_wrapper: Any = None
        if spec.family not in ("timm_cnn", "gtr"):
            # timm CNNs construct inside TimmCNNBackbone and the GTR GLA
            # line inside GTRViTBackbone — neither has a transformers
            # wrapper, so the wrapper stays None for those families.
            if spec.backbone_name.startswith("vittest"):
                backbone_wrapper = _build_vit_test_backbone()
            else:
                full_name = f"{spec.backbone_package}/{spec.backbone_name}"
                backbone_wrapper = get_wrapped_model(
                    full_name, num_input_channels=3, load_weights=load_weights
                )
        self.backbone_wrapper = backbone_wrapper

        if self._token_drop_args is not None:
            # Structural gate (docs 17 §9 / 20 §3.2): the masked forward
            # orchestrates HF internals, so the wrapper must expose a
            # supported orchestration surface — the DINOv2 layout
            # (embeddings + encoder) or the DINOv3 layout (embeddings +
            # model + rope_embeddings, doc 20). Rejecting here surfaces an
            # unsupported layout as a config error instead of a
            # mid-training failure.
            wrapper = backbone_wrapper
            if not isinstance(wrapper, TransformersDinov2Wrapper) or (
                not wrapper.supports_masked_forward()
            ):
                raise ValueError(
                    f"token_drop requires a ViT backbone whose wrapper "
                    f"exposes the masked-forward internals (DINOv2: "
                    f"embeddings + encoder; DINOv3: embeddings + model + "
                    f"rope_embeddings — design doc 20 §3.2); the wrapper "
                    f"for '{spec.backbone_name}' does not."
                )

        if spec.family == "vit":
            self.backbone: nn.Module = ViTDetectionBackbone(
                backbone_wrapper,
                interaction_indexes=spec.interaction_indexes,
                finetune=True,
                hidden_dim=hidden_dim,
                project_features=spec.project_features,
                resize_features=True,
            )
        elif spec.family == "timm_cnn":
            self.backbone = TimmCNNBackbone(
                spec.backbone_name,
                pretrained=load_weights,
                finetune=True,
                num_stages=pyramid_levels,
            )
        elif spec.family == "gtr":
            # The GTR L/X checkpoints keep a 384-wide ViT while the
            # encoder/decoder stack runs at 256; ``backbone_embed_dim``
            # separates the two widths (None keeps the shared width).
            backbone_embed_dim = spec.backbone_embed_dim or spec.hidden_dim
            self.backbone = GTRViTBackbone(
                embed_dim=backbone_embed_dim,
                # GTR head convention: dim/64 (192->3, 256->4, 384->6).
                num_heads=max(1, backbone_embed_dim // 64),
                interaction_indexes=spec.interaction_indexes,
                patch_size=spec.patch_size,
                finetune=True,
                ffn_ratio=spec.backbone_ffn_ratio,
            )
        else:
            self.backbone = ConvNeXtDetectionBackbone(
                backbone_wrapper,
                finetune=True,
                patch_size=spec.patch_size,
                num_stages=pyramid_levels,
            )
        in_channels = self.backbone.in_channels

        if backbone_weights is not None:
            self._load_backbone_weights(backbone_weights)

        # Encoder width knobs: "auto" follows the variant spec (the
        # ConvNeXt specs carry the heavy baseline recipe — 2048/1.0/1.0);
        # explicit values align with the official Dome-S "S" encoder
        # (1024/0.5/0.34), a ~4-6x lighter randomly-initialized module.
        enc_ffn = (
            spec.enc_dim_feedforward
            if enc_dim_feedforward == "auto"
            else int(enc_dim_feedforward)
        )
        enc_exp = (
            spec.enc_expansion if enc_expansion == "auto" else float(enc_expansion)
        )
        enc_depth = (
            spec.enc_depth_mult if enc_depth_mult == "auto" else float(enc_depth_mult)
        )

        if self._density_enabled:
            self.encoder = DomeHybridEncoder(
                in_channels,
                hidden_dim=hidden_dim,
                dim_feedforward=enc_ffn,
                expansion=enc_exp,
                depth_mult=enc_depth,
                use_encoder_idx=[pyramid_levels - 1],
                density=density,
            )
        elif spec.family == "gtr":
            # GTR line: GTREncoder (RF-DETR/LW-DETR style projector) instead
            # of the HybridEncoder AIFI stack — the ViTAdapter taps are
            # same-resolution and the pyramid comes from scale-factor
            # resampling (roadmap design doc 31 G2).
            from autovision.tasks.detection.shared.gtr.encoder import GTREncoder

            self.encoder = GTREncoder(
                in_channels,
                hidden_dim=hidden_dim,
                scale_factors=[2.0, 1.0, 0.5],
                feat_strides=[8, 16, 32],
                num_blocks=3,
                layer_norm=True,
                act="silu",
            )
        else:
            self.encoder = HybridEncoder(
                in_channels,
                hidden_dim=hidden_dim,
                dim_feedforward=enc_ffn,
                expansion=enc_exp,
                depth_mult=enc_depth,
                use_encoder_idx=[pyramid_levels - 1],
            )
        # Pyramid strides: 3-level [8, 16, 32]; 4-level prepends the
        # ConvNeXt stride-4 stage (Dome-faithful, design doc 13 D1).
        feat_strides = [8, 16, 32] if pyramid_levels == 3 else [4, 8, 16, 32]
        num_points = list(spec.dec_num_points)
        if len(num_points) != pyramid_levels:
            if len(set(num_points)) != 1:
                raise ValueError(
                    f"dec_num_points {num_points} cannot be expanded to "
                    f"pyramid_levels={pyramid_levels}; specify per-level "
                    f"values explicitly."
                )
            num_points = num_points[:1] * pyramid_levels
        decoder_kwargs: dict[str, Any] = dict(
            num_classes=num_classes,
            hidden_dim=decoder_hidden_dim,
            num_queries=num_queries,
            # input_proj consumes the ENCODER width; the decoder body runs
            # at decoder_hidden_dim.
            feat_channels=[hidden_dim] * pyramid_levels,
            feat_strides=feat_strides,
            num_levels=pyramid_levels,
            num_points=num_points,
            nhead=num_heads,
            num_layers=dec_layers,
            # Spec FFN width (ViT recipes 512/1792; ConvNeXt keeps the
            # rtdetrv2 component default 1024).
            dim_feedforward=spec.dec_dim_feedforward,
            num_denoising=num_denoising,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            max_gt_cap=dn_max_gt_cap,
        )
        if self.open_vocab_args is not None:
            # Open-vocab heads ride all three decoder families (design doc
            # 27 §7): contrastive per-layer score heads over the decoder's
            # persistent ``ov_text_embeds`` buffer; the v2 flags opt into
            # the text-contrastive encoder head and the layer text
            # cross-attention.
            decoder_kwargs.update(
                text_embed_dim=resolve_text_embed_dim(
                    self.open_vocab_args.text_encoder
                ),
                language_guided_selection=(
                    self.open_vocab_args.language_guided_selection
                ),
                decoder_text_attention=self.open_vocab_args.decoder_text_attention,
                text_attn_heads=self.open_vocab_args.text_attn_heads,
                ov_logit_scale_init=self.open_vocab_args.logit_scale_init,
                ov_logit_scale_max=self.open_vocab_args.logit_scale_max,
                ov_class_bias=self.open_vocab_args.use_class_bias,
            )
        if decoder == "dfine_official":
            # Official gateway-based D-FINE decoder: the architecture the
            # published baseline D-FINE COCO checkpoints were trained
            # with. Weight-compatible via detector_checkpoint;
            # density/SSL stay fork-"dfine"-only. The GTR lineage
            # (spec.dec_lineage="gtr", the gtr_s/m/l/x official assets)
            # switches the selection stage + activations to GTR's
            # GTRTransformer layout (see DFINEOfficialDecoder.gtr_compat).
            self.decoder = DFINEOfficialDecoder(
                **decoder_kwargs,
                reg_max=reg_max,
                reg_scale=reg_scale,
                activation="silu" if spec.dec_lineage == "gtr" else "relu",
                gtr_compat=(spec.dec_lineage == "gtr"),
                mask_downsample_ratio=mask_downsample_ratio,
                # The mask head bakes its output resolution (image_size /
                # downsample_ratio) and the official seg recipe is
                # fixed-shape eval (640); anchors pin to the same canvas.
                # Det mode keeps eval_spatial_size None → runtime-generated
                # anchors, exactly as before this parameter existed.
                eval_spatial_size=(
                    tuple(self.image_size) if mask_downsample_ratio is not None else None
                ),
            )
        elif decoder == "dfine":
            if self._density_enabled:
                self.decoder: nn.Module = DomeDFINEv2Decoder(
                    **decoder_kwargs,
                    reg_max=reg_max,
                    reg_scale=reg_scale,
                    min_num_select=density.resolved_min_num_select(num_queries),
                    max_num_select=density.max_num_select,
                    nms_iou_base=density.nms_iou_base,
                    nms_iou_density_scale=density.nms_iou_density_scale,
                )
            else:
                self.decoder = DFINEv2Decoder(
                    **decoder_kwargs,
                    reg_max=reg_max,
                    reg_scale=reg_scale,
                    ssl_candidate_features=self._ssl_candidates_mode,
                )
        else:
            self.decoder = RTDETRv2Decoder(**decoder_kwargs)

        # The SSL head is built LAST so its random init never shifts the
        # RNG stream of the baseline modules: with a fixed seed, an
        # ssl-enabled model and the baseline model carry IDENTICAL
        # backbone/encoder/decoder weights (bit-for-bit eval contract).
        self.ssl_head: nn.Module | None = None
        self._ssl_sigreg = ssl is not None and ssl.sigreg_coef > 0
        if ssl is not None and ssl.mode != "off":
            self.ssl_head = SSLProjectionHead(
                hidden_dim=decoder_hidden_dim,
                projection_dim=ssl.projector_dim,
                kind=ssl.projector,
                dropout=ssl.projector_dropout,
            )

        # Open-vocab vocabulary assets, built LAST (after the ssl head) so
        # no RNG draw lands inside the baseline init stream — the encoding
        # itself forks and restores the RNG (doc 27 §3.9). The frozen tower
        # lives only inside encode_class_names: the decoder's persistent
        # ``ov_text_embeds`` buffer is the sole language artifact the model
        # carries, so state_dict / EMA / optimizer / export never see a
        # text tower.
        if self.open_vocab_args is not None:
            embeds = encode_class_names(
                class_names=self.open_vocab_args.class_names,
                text_encoder=self.open_vocab_args.text_encoder,
                prompt_template=self.open_vocab_args.prompt_template,
                prompt_templates=self.open_vocab_args.prompt_templates,
                cache_embeddings=self.open_vocab_args.cache_embeddings,
            )
            self.decoder.load_ov_text_embeds(embeds)

    # ── backbone weights ────────────────────────────────────────────────
    def _load_backbone_weights(self, backbone_weights: str) -> None:
        try:
            if str(backbone_weights).endswith(".safetensors"):
                # Safetensors is the preferred source format: no pickle
                # surface at all, tensors only.
                from safetensors.torch import load_file as safetensors_load_file

                state = safetensors_load_file(str(backbone_weights), device="cpu")
            else:
                state = torch.load(
                    backbone_weights, map_location="cpu", weights_only=True
                )
        except Exception as error:
            raise CheckpointError(
                f"Failed to load backbone weights '{backbone_weights}' "
                f"under the safe loader (weights_only=True / safetensors): "
                f"{error}. Legacy pickles with non-tensor payloads are "
                f"rejected; convert the file once with torch.load(..., "
                f"weights_only=False) in a trusted context and re-save "
                f"the plain state_dict with torch.save."
            ) from error
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        target = self.backbone.backbone_model
        state = _relabel_prefix_drift(state, target)
        incompatible = target.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(
                f"backbone_weights '{backbone_weights}' contains keys unknown to "
                f"the backbone: {incompatible.unexpected_keys[:5]}..."
            )
        if incompatible.missing_keys:
            # A partial file would silently leave those tensors at their
            # random init — fail loud instead.
            raise ValueError(
                f"backbone_weights '{backbone_weights}' is missing "
                f"{len(incompatible.missing_keys)} backbone tensors (they "
                f"would stay randomly initialized): "
                f"{incompatible.missing_keys[:5]}..."
            )

    # ── three-stage contract ────────────────────────────────────────────
    def preprocess_image(self, image: Any) -> tuple[torch.Tensor, dict[str, Any]]:
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
        _, height, width = tensor.shape
        resized = TF.resize(
            tensor,
            [self.image_size[0], self.image_size[1]],
            antialias=True,
        )
        return resized, {"orig_w": width, "orig_h": height}

    def preprocess_batch(self, batch: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.normalize.mean, device=batch.device).view(-1, 1, 1)
        std = torch.tensor(self.normalize.std, device=batch.device).view(-1, 1, 1)
        return (batch - mean) / std

    def _extract_window_keep_masks(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None
    ) -> list[torch.Tensor] | None:
        """Collect per-image window keep masks from the training targets.

        Masks travel inside the targets dict (train-only; validation and
        export call this with ``targets=None`` or mask-free targets and get
        ``None`` → the dense backbone, bit-for-bit the baseline). Missing
        per-image masks pad the batch grid with ``True`` (a dense image
        keeps every window); a mask smaller than the batch grid pads with
        ``False`` (dropped windows are pure saving) — reference semantics,
        rf-detr-ssl-v2 ``lwdetr.py`` R4 transport (design doc 17 §4.3).
        """
        if not targets:
            return None
        masks = [
            target.get("window_keep_mask") if isinstance(target, dict) else None
            for target in targets
        ]
        if not any(mask is not None for mask in masks):
            return None
        if not isinstance(self.backbone, ViTDetectionBackbone):
            # Unreachable through the config gate (family == "vit"); a hard
            # guard because only that backbone's forward accepts the kwarg.
            raise ValueError(
                "window_keep_mask targets require the ViTDetectionBackbone "
                "family (design doc 17 §3.1)."
            )
        grid_h = max(
            mask.shape[0] for mask in masks if mask is not None
        )
        grid_w = max(
            mask.shape[1] for mask in masks if mask is not None
        )
        padded: list[torch.Tensor] = []
        for mask in masks:
            if mask is None:
                mask = torch.ones(grid_h, grid_w, dtype=torch.bool)
            elif mask.shape != (grid_h, grid_w):
                mask = F.pad(
                    mask,
                    (0, grid_w - mask.shape[1], 0, grid_h - mask.shape[0]),
                    value=False,
                )
            padded.append(mask.to(device=x.device))
        return padded

    def _forward_train(
        self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None
    ) -> dict:
        window_keep_masks = self._extract_window_keep_masks(x, targets)
        if window_keep_masks is not None:
            feats = list(self.backbone(x, window_keep_masks=window_keep_masks))
        else:
            feats = list(self.backbone(x))
        if self._density_enabled:
            outs, defe_info = self.encoder(feats, targets=targets)
            out = self.decoder(outs, targets=targets, defe_info=defe_info)
        else:
            outs = self.encoder(feats)
            out = self.decoder(outs, targets=targets)
        if self.ssl_head is not None:
            if self._ssl_candidates_mode:
                # Route B: the pool is the top-k candidate features. Emitted
                # in train AND eval — the contrastive loss is the only
                # objective, so the validation loss is the held-out SSL loss.
                candidate_features = out.get("ssl_candidate_features")
                if candidate_features is None:
                    raise ValueError(
                        "ssl instance_source='encoder_candidates' requires the "
                        "decoder to emit ssl_candidate_features; the SSL-only "
                        "mode requires the D-FINE decoder family "
                        "(design doc 14 §4.10)."
                    )
                if self._ssl_sigreg:
                    out["ssl_z1"], out["ssl_s1"] = self.ssl_head.forward_with_raw(
                        candidate_features
                    )
                    out["ssl_z2"], out["ssl_s2"] = self.ssl_head.forward_with_raw(
                        candidate_features
                    )
                else:
                    out["ssl_z1"] = self.ssl_head(candidate_features)
                    out["ssl_z2"] = self.ssl_head(candidate_features)
            elif self.training:
                # Two independent dropout draws through the projection head
                # give the two views of every query; the criterion selects
                # the matched rows (and the cross-view pairs under two_crop).
                query_features = out.get("query_features")
                if query_features is None:
                    raise ValueError(
                        "ssl_head is enabled but the decoder produced no "
                        "query_features; the SSL head requires the D-FINE "
                        "decoder family (design doc 14 §4.3)."
                    )
                if self._ssl_sigreg:
                    # SIGReg (design doc 16 R1-A1): also emit the RAW
                    # pre-normalization projector outputs — the unit-sphere
                    # sample can never look Gaussian, so the distribution
                    # test must see the raw projector space.
                    out["ssl_z1"], out["ssl_s1"] = self.ssl_head.forward_with_raw(
                        query_features
                    )
                    out["ssl_z2"], out["ssl_s2"] = self.ssl_head.forward_with_raw(
                        query_features
                    )
                else:
                    out["ssl_z1"] = self.ssl_head(query_features)
                    out["ssl_z2"] = self.ssl_head(query_features)
        return out

    # deploy(): inherited from TaskNetwork — it recursively calls
    # ``convert_to_deploy`` on every module, which is exactly the export
    # hook the density-guided modules implement (design doc 13 §4.8):
    # DomeHybridEncoder/DomeDFINEv2Decoder fall back to the fixed-proposal
    # baseline graph, while density-off models behave exactly as before.

    def forward_backend(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._forward_train(x)
        return out["pred_logits"], out["pred_boxes"]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_backend(x)

    def _as_output(self, raw: Any) -> ObjectDetectionOutput:
        if isinstance(raw, ObjectDetectionOutput):
            return raw
        logits, boxes = raw
        return ObjectDetectionOutput(logits=logits, boxes=boxes)

    def postprocess(
        self,
        raw_outputs: Any,
        metadata: list[dict[str, Any]],
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Shared DETR decode: normalized cxcywh -> original-pixel xyxy."""
        return decode_detection_outputs(
            raw_outputs,
            metadata,
            num_top_queries=self.num_top_queries,
            threshold=threshold,
        )

    # ── export ──────────────────────────────────────────────────────────
    @property
    def model_input_spec(self):
        from autovision.tasks.task_model_io import ModelInputSpec, TensorSpec

        return ModelInputSpec(
            input_specs={
                "images": TensorSpec(
                    shape=(3, *self.image_size), dtype=torch.float32, is_batched=True
                )
            }
        )

    def verify_onnx_export_outputs(
        self, torch_outputs: tuple, onnx_outputs: list
    ) -> None:
        torch.testing.assert_close(
            torch_outputs[0].sum(dim=1),
            torch.as_tensor(onnx_outputs[0]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )
        torch.testing.assert_close(
            torch_outputs[1].sum(dim=1),
            torch.as_tensor(onnx_outputs[1]).float().sum(dim=1),
            atol=5e-3,
            rtol=1e-1,
            equal_nan=True,
            check_device=False,
            check_dtype=False,
        )


def _dinov3_supported() -> bool:
    try:
        import transformers

        # transformers ships the classes as ``DINOv3*`` (4.x lazy exports
        # and 5.x alike); keep the ``Dinov3*`` fallbacks for older layouts.
        vit = getattr(transformers, "DINOv3ViTModel", None) or getattr(
            transformers, "Dinov3ViTModel", None
        )
        convnext = getattr(transformers, "DINOv3ConvNextModel", None) or getattr(
            transformers, "Dinov3ConvNextModel", None
        )
        return vit is not None and convnext is not None
    except ImportError:
        return False


def load_warm_start_into_network(
    network: "DETRObjectDetection", warm_start: str | Path
) -> dict[str, Any]:
    """Load an AVWarmStart v1 payload into a DETR network (doc 29 P1).

    Shape-compatible subset transfer over the payload's ``model`` state —
    the license field is mandatory (registry R7: an unlicensed warm-start
    asset must not exist) and the decoder coverage gate refuses partial
    assemblies below 0.5. Returns the load report for run provenance.

    ``warm_start`` accepts a file path or a registered user-held asset
    name (``registry/user_held_assets.py``, policy D1-a).
    """
    from autovision.registry.user_held_assets import resolve_avwarm_start_source
    from autovision.registry.warm_start import (
        AV_WARM_START_FORMAT,
        WarmStartError,
        load_warm_start,
    )

    warm_start = resolve_avwarm_start_source(warm_start)
    try:
        payload = load_warm_start(warm_start)
    except WarmStartError as error:
        raise ValueError(f"Invalid AVWarmStart asset '{warm_start}': {error}") from error
    if payload.license is None or not str(getattr(payload.license, "id", "")).strip():
        raise ValueError(
            f"AVWarmStart asset '{warm_start}' carries no license — refusing "
            "to consume it (registry R7)."
        )

    # Sampling-point split guard: the deformable-attention offsets are a
    # single flattened tensor whose shape only exposes sum(num_points) —
    # a payload assembled with a different split ((4,4,4) vs (3,6,3))
    # would transfer silently and mis-place every level's offsets.
    recorded_points = payload.spec.get("dec_num_points")
    if recorded_points is not None:
        expected_points = list(network.variant_spec.dec_num_points)
        if list(recorded_points) != expected_points:
            raise ValueError(
                f"AVWarmStart '{warm_start}' was assembled with "
                f"num_points={list(recorded_points)} but the network's "
                f"variant expects {expected_points} — refusing a "
                "semantic-mismatch transfer (the sum-only shape identity "
                "makes this undetectable downstream)."
            )
    recorded_lineage = payload.spec.get("dec_lineage")
    if (
        recorded_lineage is not None
        and recorded_lineage != network.variant_spec.dec_lineage
    ):
        raise ValueError(
            f"AVWarmStart '{warm_start}' dec_lineage '{recorded_lineage}' "
            f"does not match '{network.variant_spec.dec_lineage}' — the two "
            "lineages decode with different selection-stage semantics."
        )

    sta_keys = sorted(k for k in payload.model if k.startswith("backbone.sta."))
    if sta_keys:
        raise ValueError(
            f"AVWarmStart '{warm_start}' contains {len(sta_keys)} "
            "backbone.sta.* tensor(s) from the DEIMv2-derived spatial prior, "
            "which was removed for licensing. The no-STA ViT adapter cannot "
            "load this checkpoint faithfully; retrain/export a no-STA "
            "checkpoint or use a ConvNeXt asset."
        )

    target = network.state_dict()
    transferred: dict[str, torch.Tensor] = {}
    skipped_shape: list[str] = []
    for key, tensor in payload.model.items():
        if key not in target:
            continue
        if tensor.shape != target[key].shape:
            skipped_shape.append(key)
            continue
        transferred[key] = tensor.to(dtype=target[key].dtype)
    missing, unexpected = network.load_state_dict(transferred, strict=False)

    decoder_keys = [k for k in target if k.startswith("decoder.")]
    decoder_total = sum(target[k].numel() for k in decoder_keys)
    decoder_done = sum(
        target[k].numel() for k in transferred if k in decoder_keys
    )
    coverage = decoder_done / decoder_total if decoder_total else 0.0
    if coverage < 0.5:
        raise ValueError(
            f"AVWarmStart '{warm_start}' decoder coverage {coverage:.3f} < 0.5 "
            "— refusing a partial warm-start."
        )
    return {
        "format": AV_WARM_START_FORMAT,
        "license": str(payload.license.id),
        "transferred": len(transferred),
        "skipped_shape": len(skipped_shape),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "decoder_coverage": round(coverage, 4),
    }
