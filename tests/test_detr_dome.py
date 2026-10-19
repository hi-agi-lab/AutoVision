# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the DETR density-guided branch (Dome-DETR port, doc 13 §7).

All cases run offline against the tiny ``dinov2-vittest-notpretrained``
variant (hermetic, CPU-friendly), mirroring the conventions of
``tests/test_detr_object_detection.py``.
"""

import math
import time

import pytest
import torch
from torch import nn

from autovision.tasks.detection.detr import (
    DETRDensityArgs,
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
)
from autovision.tasks.detection.detr.dome_decoder import (
    _PADDED_LOGIT,
    DomeDFINEv2Decoder,
)
from autovision.tasks.detection.detr.dome_encoder import (
    DomeHybridEncoder,
)
from autovision.tasks.detection.shared.defe import (
    GaussHeatmapGenerator,
    LiteDeFE,
    defe_count_loss,
    defe_density_loss,
)
from autovision.tasks.detection.shared.dynamic_nms import (
    per_class_dynamic_nms,
)
from autovision.tasks.detection.shared.mwas import (
    WindowProcessor,
    adaptive_window_mask,
    window_grid_size,
)

PADDED_LOGIT = -1e4


def make_density_args(**overrides) -> DETRDensityArgs:
    defaults: dict = dict(max_num_select=50, mwas_window_size=5)
    defaults.update(overrides)
    return DETRDensityArgs(**defaults)


def make_dome_train_model(**density_overrides) -> DETRObjectDetectionTrain:
    args = DETRObjectDetectionTrainArgs(
        num_classes=2,
        backbone="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        num_queries=20,
        num_top_queries=20,
        decoder="dfine",
        density=make_density_args(**density_overrides),
    )
    return DETRObjectDetectionTrain(args)


def make_baseline_train_model() -> DETRObjectDetectionTrain:
    args = DETRObjectDetectionTrainArgs(
        num_classes=2,
        backbone="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        num_queries=20,
        num_top_queries=20,
        decoder="dfine",
    )
    return DETRObjectDetectionTrain(args)


def make_batch(images: int = 2, size: int = 224):
    return {
        "image": torch.randn(images, 3, size, size),
        "bboxes": [
            torch.rand(4, 4) * 0.3 + 0.3,
            torch.rand(4, 4) * 0.3 + 0.3,
        ],
        "classes": [
            torch.tensor([0, 1, 0, 1]),
            torch.tensor([0, 1, 0, 1]),
        ],
    }


def run_training_step(train_model, batch):
    return train_model.training_step(
        {
            "image": batch["image"],
            "bboxes": batch["bboxes"],
            "classes": batch["classes"],
        },
        0,
    )


class _StubDensity(nn.Module):
    """Replaces LiteDeFE with a crafted density map for gating tests."""

    def __init__(self, density_map: torch.Tensor) -> None:
        super().__init__()
        self.density_map = density_map

    def forward(self, features):
        reg_value = torch.zeros(features.shape[0], 1, device=features.device)
        return self.density_map.to(features.device), reg_value


class TestDensityArgs:
    def test_resolved_min_num_select(self):
        assert make_density_args().resolved_min_num_select(20) == 20
        assert make_density_args(min_num_select=30).resolved_min_num_select(20) == 30

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValueError):
            DETRDensityArgs(unknown_field=1)


class TestDeFE:
    def test_output_shapes_range_and_normalization(self):
        defe = LiteDeFE(64)
        features = torch.randn(2, 64, 28, 28)
        density, reg_value = defe(features)
        # stride 8 in -> stride 2 out (224 -> 112); D1.
        assert density.shape == (2, 1, 112, 112)
        assert (density >= 0).all() and (density <= 1).all()
        assert torch.allclose(density.amax(dim=(-2, -1)), torch.ones(2))
        assert reg_value.shape == (2, 1)
        assert (reg_value >= 0).all() and (reg_value <= 1).all()

    def test_backward_through_losses(self):
        defe = LiteDeFE(64)
        features = torch.randn(1, 64, 28, 28)
        density, reg_value = defe(features)
        gt = GaussHeatmapGenerator((112, 112))(torch.tensor([[0.5, 0.5, 0.1, 0.1]]))
        loss = defe_density_loss(density, gt) + defe_count_loss(
            reg_value, [3], min_num_select=10, max_num_select=50
        )
        loss.backward()
        grads = [
            p.grad.abs().sum() for p in defe.parameters() if p.grad is not None
        ]
        assert grads and float(sum(grads)) > 0

    def test_heatmap_peak_at_box_center(self):
        generator = GaussHeatmapGenerator((112, 112), sigma_ratio=1.2)
        boxes = torch.tensor([[0.5, 0.5, 0.1, 0.1]])
        heatmap = generator(boxes)
        assert heatmap.shape == (1, 112, 112)
        peak = heatmap[0].argmax()
        peak_y, peak_x = divmod(int(peak), 112)
        assert abs(peak_y - 56) <= 1 and abs(peak_x - 56) <= 1
        assert float(heatmap.max()) == pytest.approx(1.0)

    def test_heatmap_empty_boxes(self):
        generator = GaussHeatmapGenerator((64, 64))
        heatmap = generator(torch.zeros(0, 4))
        assert heatmap.shape == (1, 64, 64)
        assert float(heatmap.max()) == 0.0

    def test_heatmap_two_boxes_normalized(self):
        generator = GaussHeatmapGenerator((112, 112))
        boxes = torch.tensor([[0.25, 0.25, 0.1, 0.1], [0.75, 0.75, 0.2, 0.2]])
        heatmap = generator(boxes)
        assert float(heatmap.max()) == pytest.approx(1.0)
        # Both centers carry mass.
        assert float(heatmap[0, 28, 28]) > 0.1
        assert float(heatmap[0, 84, 84]) > 0.1


class TestDensityLosses:
    def test_density_loss_underestimation_penalized(self):
        target = torch.full((1, 1, 8, 8), 0.5)
        under = torch.full((1, 1, 8, 8), 0.4)
        over = torch.full((1, 1, 8, 8), 0.6)
        loss_under = defe_density_loss(under, target)
        loss_over = defe_density_loss(over, target)
        assert loss_under > loss_over > 0

    def test_density_loss_perfect_prediction_is_zero(self):
        target = torch.rand(1, 1, 8, 8)
        assert float(defe_density_loss(target, target)) == pytest.approx(0.0)

    def test_count_loss_asymmetry(self):
        target_count = 30  # ratio (30 - 10) / 40 = 0.5
        under = torch.tensor([[0.2]])
        over = torch.tensor([[0.8]])
        loss_under = defe_count_loss(
            under, [target_count], min_num_select=10, max_num_select=50
        )
        loss_over = defe_count_loss(
            over, [target_count], min_num_select=10, max_num_select=50
        )
        assert loss_under > loss_over > 0

    def test_count_loss_exact_target_is_zero(self):
        # n_gt == min_num_select -> target ratio 0; pred 0 -> zero loss.
        loss = defe_count_loss(
            torch.tensor([[0.0]]), [10], min_num_select=10, max_num_select=50
        )
        assert float(loss) == pytest.approx(0.0)

    def test_count_loss_clamps_extreme_counts(self):
        # n_gt beyond both bounds clamps to max -> target 1.0.
        loss = defe_count_loss(
            torch.tensor([[1.0]]), [10_000], min_num_select=10, max_num_select=50
        )
        assert float(loss) == pytest.approx(0.0)


class TestDynamicNMS:
    def test_suppresses_duplicate_same_class(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5], [50.0, 50.0, 60.0, 60.0]]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        class_ids = torch.zeros(3, dtype=torch.long)
        low_threshold = torch.full((3,), 0.1)
        keep = per_class_dynamic_nms(boxes, scores, class_ids, low_threshold)
        assert keep.tolist() == [0, 2]

        high_threshold = torch.full((3,), 0.9)
        keep = per_class_dynamic_nms(boxes, scores, class_ids, high_threshold)
        assert keep.tolist() == [0, 1, 2]

    def test_per_class_independence(self):
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]]
        )
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.tensor([0, 1])
        thresholds = torch.full((2,), 0.1)
        keep = per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
        assert keep.tolist() == [0, 1]

    def test_threshold_uses_kept_box(self):
        # IoU 0.68 pair: kept-box threshold decides suppression.
        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0]]
        )
        scores = torch.tensor([0.9, 0.8])
        class_ids = torch.zeros(2, dtype=torch.long)
        keep = per_class_dynamic_nms(
            boxes, scores, class_ids, torch.tensor([0.7, 0.7])
        )
        assert keep.tolist() == [0, 1]
        keep = per_class_dynamic_nms(
            boxes, scores, class_ids, torch.tensor([0.6, 0.6])
        )
        assert keep.tolist() == [0]

    def test_empty_input(self):
        keep = per_class_dynamic_nms(
            torch.zeros(0, 4),
            torch.zeros(0),
            torch.zeros(0, dtype=torch.long),
            torch.zeros(0),
        )
        assert keep.numel() == 0


class TestMWAS:
    def test_window_grid_size(self):
        assert window_grid_size(14, 14, 5) == (3, 3)
        assert window_grid_size(50, 50, 10) == (5, 5)
        assert window_grid_size(36, 36, 10) == (4, 4)

    def test_adaptive_window_mask_always_nonempty(self):
        pooled = torch.zeros(2, 1, 3, 3)
        mask = adaptive_window_mask(pooled)
        assert mask.dtype == torch.bool
        assert mask.sum(dim=(1, 2)).min() >= 1

        pooled = torch.zeros(1, 1, 3, 3)
        pooled[0, 0, 2, 2] = 0.5
        mask = adaptive_window_mask(pooled, threshold=0.05)
        assert mask[0, 2, 2]

    def test_window_processor_residual_locality(self):
        torch.manual_seed(0)
        processor = WindowProcessor(embed_dim=64, dim_feedforward=128)
        memory = torch.randn(1, 64, 14, 14)
        mask = torch.zeros(1, 3, 3, dtype=torch.bool)
        mask[0, 0, 0] = True
        glob_pos = torch.randn(64, 14, 14)
        out = processor(memory, mask, 5, glob_pos)
        assert out.shape == memory.shape

        residual = (out - memory).abs().sum(dim=1).detach()
        # Only the selected window's 5x5 block changes.
        assert float(residual[:5, :5].sum()) > 0
        rest = residual.clone()
        rest[:5, :5] = 0
        assert float(rest.max()) == 0.0

    def test_window_processor_multiple_windows(self):
        torch.manual_seed(0)
        processor = WindowProcessor(embed_dim=32, dim_feedforward=64)
        memory = torch.randn(2, 32, 14, 14)
        mask = torch.ones(2, 3, 3, dtype=torch.bool)
        out = processor(memory, mask, 5, torch.randn(32, 14, 14))
        assert out.shape == memory.shape
        assert not torch.allclose(out, memory)


class TestDenoisingDynamicMask:
    @staticmethod
    def _targets(device="cpu"):
        return [
            {
                "labels": torch.tensor([0, 1], device=device),
                "boxes": torch.rand(2, 4, device=device) * 0.3 + 0.3,
            },
            {
                "labels": torch.tensor([1], device=device),
                "boxes": torch.rand(1, 4, device=device) * 0.3 + 0.3,
            },
        ]

    def _group(self, batch_queries_num=None, num_heads=None, num_queries=10):
        from autovision.tasks.detection.shared.denoising import (
            get_contrastive_denoising_training_group,
        )

        embed = nn.Embedding(3, 16, padding_idx=2)
        return get_contrastive_denoising_training_group(
            self._targets(),
            num_classes=2,
            num_queries=num_queries,
            class_embed=embed,
            num_denoising=4,
            batch_queries_num=batch_queries_num,
            num_heads=num_heads,
        )

    def test_baseline_path_still_2d(self):
        _, _, attn_mask, dn_meta = self._group()
        assert attn_mask.dim() == 2
        # num_denoising=4, max_gt=2 -> num_group=2 -> 8 dn queries.
        assert dn_meta["dn_num_split"] == [8, 10]

    def test_no_padding_keeps_2d(self):
        _, _, attn_mask, _ = self._group(batch_queries_num=[10, 10], num_heads=2)
        assert attn_mask.dim() == 2

    def test_padding_expands_to_3d_and_blocks_leakage(self):
        _, _, attn_mask, _ = self._group(batch_queries_num=[10, 5], num_heads=2)
        assert attn_mask.dim() == 3
        assert attn_mask.shape[0] == 4  # 2 images x 2 heads
        # Image 1 (heads 2:4): 8 dn + 5 valid queries; padded tail must be
        # mutually invisible with the valid prefix, visible to itself.
        for head in (2, 3):
            block = attn_mask[head]
            assert block[:13, 13:].all()  # valid -> padding blocked
            assert block[13:, :13].all()  # padding -> valid blocked
            assert not block[13:, 13:].any()  # padding sees itself
            assert attn_mask[0].shape == block.shape

    def test_forward_no_nan_with_padding_mask(self):
        _, _, attn_mask, _ = self._group(batch_queries_num=[10, 5], num_heads=2)
        # 3D mask leading dim = batch * heads = 2 * 2.
        layer = nn.MultiheadAttention(16, 2, batch_first=True)
        query = torch.randn(2, 18, 16)
        out, _ = layer(query, query, query, attn_mask=attn_mask)
        assert torch.isfinite(out).all()


class TestDomeTaskModel:
    def test_density_off_constructs_baseline_modules(self):
        train_model = make_baseline_train_model()
        from autovision.tasks.detection.shared.dfine_decoder import (
            DFINEv2Decoder,
        )
        from autovision.tasks.detection.shared.hybrid_encoder import (
            HybridEncoder,
        )

        assert type(train_model.model.encoder) is HybridEncoder
        assert type(train_model.model.decoder) is DFINEv2Decoder

    def test_density_on_constructs_dome_modules(self):
        train_model = make_dome_train_model()
        assert type(train_model.model.encoder).__name__ == "DomeHybridEncoder"
        assert type(train_model.model.decoder).__name__ == "DomeDFINEv2Decoder"

    def test_density_requires_dfine_decoder(self):
        args = DETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="dinov2-vittest-notpretrained",
            image_size=(224, 224),
            density=make_density_args(),
            decoder="rtdetrv2",
        )
        with pytest.raises(ValueError, match="requires the D-FINE decoder"):
            DETRObjectDetectionTrain(args)

    def test_pyramid4_requires_convnext_family(self):
        from autovision.tasks.detection.detr.task_model import (
            DETRObjectDetection,
        )

        with pytest.raises(ValueError, match="ConvNeXt-family"):
            DETRObjectDetection(
                num_classes=2,
                model_name="dinov2-vittest-notpretrained",
                image_size=(224, 224),
                pyramid_levels=4,
            )
        with pytest.raises(ValueError, match="pyramid_levels must be 3 or 4"):
            DETRObjectDetection(
                num_classes=2,
                model_name="dinov2-vittest-notpretrained",
                image_size=(224, 224),
                pyramid_levels=5,
            )

    def test_pyramid4_shapes_with_density(self):
        """Dome-faithful 4-level pyramid: DeFE lands at stride 4 (density
        map at full resolution), MWAS at stride 8, AIFI on the deepest
        level, and the density-guided decoder consumes 4 levels. Built
        from a local tiny DINOv3-ConvNeXt (hermetic, no HF download)."""
        import transformers

        from autovision.encoders.dinov3.dinov3_wrapper import (
            TransformersDinov3ConvNextWrapper,
        )
        from autovision.tasks.detection.detr.backbone_wrappers import (
            ConvNeXtDetectionBackbone,
        )
        from autovision.tasks.detection.detr.dome_decoder import (
            DomeDFINEv2Decoder,
        )

        hf_config = getattr(
            transformers, "DINOv3ConvNextConfig", None
        ) or getattr(transformers, "Dinov3ConvNextConfig")
        hf_model_cls = getattr(
            transformers, "DINOv3ConvNextModel", None
        ) or getattr(transformers, "Dinov3ConvNextModel")
        wrapper = TransformersDinov3ConvNextWrapper(
            hf_model_cls(
                hf_config(
                    hidden_sizes=[8, 16, 32, 64],
                    depths=[1, 1, 1, 1],
                    num_channels=3,
                )
            )
        )
        backbone = ConvNeXtDetectionBackbone(wrapper, num_stages=4)
        assert backbone.num_stages == 4
        assert backbone.in_channels == [8, 16, 32, 64]

        hidden = 64
        encoder = DomeHybridEncoder(
            [8, 16, 32, 64],
            hidden_dim=hidden,
            dim_feedforward=128,
            use_encoder_idx=[3],
            density=make_density_args(max_num_select=50, mwas_window_size=5),
        )
        decoder = DomeDFINEv2Decoder(
            num_classes=2,
            hidden_dim=hidden,
            num_queries=20,
            feat_channels=[hidden] * 4,
            feat_strides=[4, 8, 16, 32],
            num_levels=4,
            nhead=4,
            num_layers=2,
            num_denoising=10,
            reg_max=8,
            min_num_select=20,
            max_num_select=50,
        )
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            feats = list(backbone(torch.randn(1, 3, 224, 224)))
            assert [tuple(f.shape[-2:]) for f in feats] == [
                (56, 56), (28, 28), (14, 14), (7, 7),
            ]
            outs, info = encoder(feats)
            assert len(outs) == 4
            # Density map at full resolution (stride 1 — D1 at 4 levels).
            assert tuple(info["density_map"].shape[-2:]) == (224, 224)
            # MWAS windows on the stride-8 level: ceil(28/5) = 6.
            assert tuple(info["window_mask"].shape[-2:]) == (6, 6)
            out = decoder(outs, defe_info=info)
        assert out["pred_logits"].shape[-1] == 2
        assert len(out["batch_queries_num"]) == 1
        assert all(20 <= n <= 50 for n in out["batch_queries_num"])

    def test_num_queries_must_fit_max_select(self):
        args = DETRObjectDetectionTrainArgs(
            num_classes=2,
            backbone="dinov2-vittest-notpretrained",
            image_size=(224, 224),
            num_queries=60,
            density=make_density_args(max_num_select=50),
            decoder="dfine",
        )
        with pytest.raises(ValueError, match="max_num_select"):
            DETRObjectDetectionTrain(args)

    def test_training_step_losses_and_backward(self):
        train_model = make_dome_train_model()
        batch = make_batch()
        result = run_training_step(train_model, batch)
        assert torch.isfinite(result.loss)
        for key in ("defe_density_loss", "defe_count_loss"):
            assert key in result.log_dict
            assert math.isfinite(result.log_dict[key])
        result.loss.backward()
        dome_params = dict(train_model.model.named_parameters())
        for prefix in ("encoder.defe", "encoder.mwas_processor"):
            grads = [
                p.grad.abs().sum()
                for name, p in dome_params.items()
                if name.startswith(prefix) and p.grad is not None
            ]
            assert grads and float(sum(grads)) > 0, prefix

    def test_density_off_has_no_dome_keys(self):
        baseline = make_baseline_train_model().model.state_dict().keys()
        dome = make_dome_train_model().model.state_dict().keys()
        assert not any("defe" in k or "mwas" in k for k in baseline)
        assert any("defe" in k for k in dome)

    def test_variable_query_selection_and_padding(self):
        train_model = make_dome_train_model()
        model = train_model.model
        model.eval()

        with torch.no_grad():
            feats = list(model.backbone(torch.randn(2, 3, 224, 224)))
            outs, defe_info = model.encoder(feats)
            # Image 0: dense everywhere. Image 1: one hot spot -> a single
            # selected window -> far fewer surviving candidates.
            crafted = torch.full((2, 1, 112, 112), 0.5)
            crafted[1] = 0.0
            crafted[1, 0, :20, :20] = 1.0
            defe_info["density_pooled"] = torch.nn.functional.adaptive_max_pool2d(
                crafted, (3, 3)
            )
            defe_info["window_mask"] = adaptive_window_mask(
                defe_info["density_pooled"]
            )
            assert defe_info["window_mask"][0].sum() == 9
            assert int(defe_info["window_mask"][1].sum()) < 9
            out = model.decoder(outs, defe_info=defe_info)

        queries = out["batch_queries_num"]
        assert len(queries) == 2
        assert all(20 <= n <= 50 for n in queries)
        assert queries[0] == 50
        assert queries[1] < queries[0]

        valid = out["query_valid_mask"]
        assert valid[:, : queries[1]].all()
        assert not valid[1, queries[1] :].any()
        # D5: padded-query logits buried at the source.
        assert (out["pred_logits"][1, queries[1] :] <= PADDED_LOGIT).all()
        # Valid-query logits stay above the fill value.
        assert (out["pred_logits"][1, : queries[1]] > PADDED_LOGIT).all()

    def test_decode_ignores_padded_queries(self):
        train_model = make_dome_train_model()
        model = train_model.model
        model.eval()
        with torch.no_grad():
            out = model._forward_train(torch.randn(1, 3, 224, 224))
        decoded = model.postprocess(
            (out["pred_logits"], out["pred_boxes"]),
            [{"orig_w": 224, "orig_h": 224}],
            threshold=0.5,
        )
        assert len(decoded) == 1
        assert decoded[0]["scores"].numel() <= out["batch_queries_num"][0]

    def test_training_with_padded_queries_and_zero_gt(self):
        train_model = make_dome_train_model()
        batch = {
            "image": torch.randn(2, 3, 224, 224),
            "bboxes": [torch.rand(6, 4) * 0.3 + 0.3, torch.zeros(0, 4)],
            "classes": [torch.tensor([0, 1, 0, 1, 0, 1]), torch.zeros(0, dtype=torch.long)],
        }
        result = run_training_step(train_model, batch)
        assert torch.isfinite(result.loss)
        result.loss.backward()

    def test_deploy_static_falls_back_to_fixed_queries(self):
        train_model = make_dome_train_model()
        model = train_model.model
        # TaskNetwork.deploy() recursively invokes convert_to_deploy; the
        # dome modules participate through that hook (design doc 13 §4.8).
        model.deploy()
        assert model.encoder._static_export
        assert model.decoder._static_export
        model.eval()
        with torch.no_grad():
            out = model._forward_train(torch.randn(2, 3, 224, 224))
        # Fixed top-min_num_select (== num_queries == 20), no density keys.
        assert out["pred_logits"].shape == (2, 20, 2)
        assert "batch_queries_num" not in out

    def test_deploy_baseline_follows_base_contract(self):
        model = make_baseline_train_model().model
        model.deploy()
        # No dome state exists on the baseline modules.
        assert not getattr(model.encoder, "_static_export", False)
        # The pre-existing export contract ran: RepVgg blocks are fused.
        repvggs = [
            m for m in model.encoder.modules() if type(m).__name__ == "RepVggBlock"
        ]
        assert repvggs and all(hasattr(m, "conv") for m in repvggs)
        model.eval()
        with torch.no_grad():
            out = model._forward_train(torch.randn(1, 3, 224, 224))
        assert out["pred_logits"].shape == (1, 20, 2)

    def test_checkpoint_roundtrip(self):
        train_model = make_dome_train_model()
        batch = make_batch()
        result = run_training_step(train_model, batch)
        result.loss.backward()
        state = train_model.model.state_dict()

        fresh = make_dome_train_model()
        fresh.model.load_state_dict(state)
        fresh.model.eval()
        train_model.model.eval()
        with torch.no_grad():
            expected = train_model.model._forward_train(batch["image"])
            actual = fresh.model._forward_train(batch["image"])
        assert torch.allclose(expected["pred_logits"], actual["pred_logits"])
        assert torch.allclose(expected["pred_boxes"], actual["pred_boxes"])
        assert expected["batch_queries_num"] == actual["batch_queries_num"]

    def test_validation_step_logs_dome_metrics(self):
        train_model = make_dome_train_model()
        train_model.set_categories({0: "cat0", 1: "cat1"})
        batch = make_batch()
        batch["original_size"] = [(100, 80), (100, 80)]
        result = train_model.validation_step(batch, 0)
        assert torch.isfinite(result.loss)
        assert "dome_avg_queries" in result.log_dict
        assert "dome_ample_rate" in result.log_dict
        assert 0.0 <= result.log_dict["dome_ample_rate"] <= 1.0
        assert result.log_dict["dome_avg_queries"] >= 20
        # The engine only persists compute_metrics output — the dome
        # diagnostics must surface there too (design doc 13 §4.6).
        metrics = train_model.compute_metrics("val")
        assert "val_metric/dome_avg_queries" in metrics
        assert "val_metric/dome_ample_rate" in metrics
        # Gate-tightness readout: the adaptive window mask always selects
        # at least one window per image, so the ratio lies in (0, 1].
        assert "val_metric/dome_window_ratio" in metrics
        assert 0.0 < metrics["val_metric/dome_window_ratio"] <= 1.0
        consumed = train_model.compute_metrics("val")
        assert "val_metric/dome_avg_queries" not in consumed
        assert "val_metric/dome_window_ratio" not in consumed

    def test_window_ratio_rides_the_ddp_metric_state(self):
        train_model = make_dome_train_model()
        train_model.set_categories({0: "cat0", 1: "cat1"})
        batch = make_batch()
        batch["original_size"] = [(100, 80), (100, 80)]
        train_model.validation_step(batch, 0)
        state = train_model.export_metric_state("val")
        assert state["dome_window_ratios"]
        train_model.reset_metrics("val")
        assert train_model._val_window_ratios == []
        train_model.merge_metric_state("val", state)
        metrics = train_model.compute_metrics("val")
        assert 0.0 < metrics["val_metric/dome_window_ratio"] <= 1.0

    def test_validation_report_footer_follows_iou_term(self):
        # The stale-GIoU footgun: with box_regression_iou="siou" (doc 15)
        # the note must name SIoU, the default must keep GIoU.
        baseline = make_baseline_train_model()
        report = baseline.validation_report({"val_metric/map": 0.0})
        assert "VFL/bbox/GIoU only" in report
        siou_train = DETRObjectDetectionTrain(
            DETRObjectDetectionTrainArgs(
                num_classes=2,
                backbone="dinov2-vittest-notpretrained",
                image_size=(224, 224),
                num_queries=20,
                num_top_queries=20,
                decoder="dfine",
                box_regression_iou="siou",
            )
        )
        siou_report = siou_train.validation_report({"val_metric/map": 0.0})
        assert "VFL/bbox/SIoU only" in siou_report

    def test_validation_step_baseline_has_no_dome_metrics(self):
        train_model = make_baseline_train_model()
        train_model.set_categories({0: "cat0", 1: "cat1"})
        batch = make_batch()
        batch["original_size"] = [(100, 80), (100, 80)]
        result = train_model.validation_step(batch, 0)
        assert "dome_avg_queries" not in result.log_dict

    def test_optimizer_covers_dome_params(self):
        train_model = make_dome_train_model()
        optimizer, _ = train_model.get_optimizer(
            total_steps=100,
            global_batch_size=16,
            train_num_batches=100,
        )
        grouped = {
            id(p)
            for group in optimizer.param_groups
            for p in group["params"]
        }
        dome_params = dict(train_model.model.named_parameters())
        for name, param in dome_params.items():
            if name.startswith(("encoder.defe", "encoder.mwas_processor")):
                assert id(param) in grouped, name

    def test_export_weights_smoke(self):
        pytest.importorskip("torch", minversion="2.0")
        train_model = make_dome_train_model()
        state = train_model.get_export_state_dict()
        assert any("defe" in key for key in state)

    def test_init_args_snapshot_is_safe_loader_friendly(self):
        # Exported checkpoints are reloaded with weights_only=True; the
        # density config must ride as a plain dict (design doc 13 §4.9)
        # and reconstruct an identical model.
        model = make_dome_train_model().model
        init_args = model.init_args
        assert isinstance(init_args["density"], dict)
        rebuilt = type(model)(**init_args)
        assert set(rebuilt.state_dict().keys()) == set(model.state_dict().keys())


class TestDomeDecoderUnit:
    """Direct decoder-level checks that need no full task model."""

    @staticmethod
    def _decoder(**overrides) -> DomeDFINEv2Decoder:
        kwargs = dict(
            num_classes=2,
            hidden_dim=64,
            num_queries=10,
            feat_channels=[64, 64, 64],
            feat_strides=[8, 16, 32],
            num_levels=3,
            nhead=4,
            num_layers=2,
            num_denoising=10,
            reg_max=8,
            min_num_select=10,
            max_num_select=30,
        )
        kwargs.update(overrides)
        return DomeDFINEv2Decoder(**kwargs)

    @staticmethod
    def _defe_info(batch=2, n_h=3, n_w=3, pooled_fill=1.0):
        pooled = torch.full((batch, 1, n_h, n_w), pooled_fill)
        window_mask = adaptive_window_mask(pooled)
        return {
            "density_pooled": pooled,
            "window_mask": window_mask,
            "reg_value": torch.zeros(batch, 1),
        }

    def test_min_max_validation(self):
        with pytest.raises(ValueError, match="min_num_select"):
            self._decoder(min_num_select=0, max_num_select=0)
        with pytest.raises(ValueError, match="min_num_select"):
            self._decoder(min_num_select=40, max_num_select=30)

    def test_queries_bounded_by_pool_size(self):
        decoder = self._decoder()
        # Fewer total cells than max_num_select -> selection capped; the
        # all-dense pooled map keeps every window so nothing is gated.
        spatial_shapes = [(5, 8), (4, 8), (2, 8)]
        total_cells = sum(h * w for h, w in spatial_shapes)
        memory = torch.randn(1, total_cells, 64)
        info = self._defe_info(batch=1)
        (
            content,
            coords,
            queries,
            valid,
            enc_logits,
            enc_boxes,
        ) = decoder._get_decoder_input_dynamic(memory, spatial_shapes, info)
        expected = min(30, total_cells)
        assert queries == [expected]
        assert content.shape == (1, expected, 64)
        assert coords.shape == (1, expected, 4)
        assert valid.all()
        # Encoder-proposal supervision payload: valid rows carry the real
        # logits/boxes; there are no padded rows in this single-image batch.
        assert enc_logits.shape == (1, expected, decoder.num_classes)
        assert enc_boxes.shape == (1, expected, 4)
        assert (enc_boxes <= 1.0).all() and (enc_boxes >= 0.0).all()
        assert torch.isfinite(enc_logits).all()

    def test_enc_supervision_rows_masked_when_padded(self):
        # Batch with unequal query counts: the padded rows of the enc
        # supervision payload must be buried at _PADDED_LOGIT and their
        # boxes zeroed, so the matcher never assigns them.
        decoder = self._decoder()
        spatial_shapes = [(5, 8), (4, 8), (2, 8)]
        total_cells = sum(h * w for h, w in spatial_shapes)
        memory = torch.randn(2, total_cells, 64)
        info = self._defe_info(batch=2)
        (
            content,
            coords,
            queries,
            valid,
            enc_logits,
            enc_boxes,
        ) = decoder._get_decoder_input_dynamic(memory, spatial_shapes, info)
        padded = max(queries)
        assert len(queries) == 2 and padded >= max(queries)
        for b, n_b in enumerate(queries):
            assert torch.isfinite(enc_logits[b, :n_b]).all()
            assert (enc_logits[b, n_b:] == _PADDED_LOGIT).all()
            assert (enc_boxes[b, n_b:] == 0).all()
        assert valid.sum() == sum(queries)

    def test_deploy_falls_back_to_baseline_forward(self):
        decoder = self._decoder()
        decoder.eval()
        decoder.convert_to_deploy()
        feats = [torch.randn(1, 64, 28, 28), torch.randn(1, 64, 14, 14),
                 torch.randn(1, 64, 7, 7)]
        with torch.no_grad():
            out = decoder(feats)
        assert out["pred_logits"].shape == (1, 10, 2)
        assert "defe_info" not in out


class TestDynamicNmsPerformance:
    """Problem-1 follow-up: the greedy scan must stay off the GPU sync
    path (a torch-loop regression costs ~0.5-1 s/step at VisDrone query
    counts)."""

    @staticmethod
    def _reference_greedy(boxes, scores, thresholds):
        """Deliberately naive O(n^2) greedy NMS in pure Python."""

        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            union = area_a + area_b - inter
            return inter / union if union > 0 else 0.0

        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        suppressed = [False] * len(scores)
        kept = []
        for i in order:
            if suppressed[i]:
                continue
            kept.append(i)
            for j in order:
                if j == i or suppressed[j]:
                    continue
                if iou(boxes[i], boxes[j]) >= thresholds[i]:
                    suppressed[j] = True
        return sorted(kept)

    def test_matches_naive_reference(self):
        import random

        rng = random.Random(0)
        for _ in range(5):
            n = 120
            boxes = torch.tensor(
                [
                    [
                        rng.uniform(0, 90),
                        rng.uniform(0, 90),
                        0.0,
                        0.0,
                    ]
                    for _ in range(n)
                ]
            )
            # Build proper xyxy with a few near-duplicates per cluster.
            boxes[:, 2:] = boxes[:, :2] + torch.tensor([10.0, 10.0]) + 20 * torch.rand(n, 2)
            scores = torch.rand(n)
            class_ids = torch.randint(0, 5, (n,))
            thresholds = 0.3 + 0.6 * torch.rand(n)

            keep = per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
            reference_by_class = set()
            for c in class_ids.unique().tolist():
                idx = [i for i in range(n) if int(class_ids[i]) == c]
                sub_boxes = boxes[idx].tolist()
                sub_scores = [float(s) for s in scores[idx]]
                sub_thr = [float(t) for t in thresholds[idx]]
                for local in self._reference_greedy(sub_boxes, sub_scores, sub_thr):
                    reference_by_class.add(idx[local])
            assert sorted(keep.tolist()) == sorted(reference_by_class)

    def test_realistic_batch_is_fast(self):
        # 500 candidates / 10 classes (VisDrone recipe) — the numpy scan
        # must stay orders of magnitude below the old ~0.2 s GPU loop.
        boxes = torch.rand(500, 4) * 50
        boxes[:, 2:] += boxes[:, :2] + 1.0
        scores = torch.rand(500)
        class_ids = torch.randint(0, 10, (500,))
        thresholds = torch.full((500,), 0.5)
        start = time.perf_counter()
        per_class_dynamic_nms(boxes, scores, class_ids, thresholds)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.5, f"dynamic NMS regressed: {elapsed:.3f}s"
