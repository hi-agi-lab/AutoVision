# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the box-aware background eraser (rf-detr-ssl-v2 port).

Contracts: protected cells are never erased, the per-box ``min_keep``
interior floor keeps at least that many patches per object, tiny boxes
stay fully protected, the dilation ring survives interior mode, and the
transform integration is off by default and absent from the val recipe.
"""

from __future__ import annotations

import torch
from torchvision import tv_tensors

from autovision.augment.detection import (
    RTDETRDetectionTransform,
    RTDETRDetectionTransformArgs,
    build_val_transform_args,
)
from autovision.augment.detection.box_aware_eraser import (
    erase_background,
    protected_cell_mask,
    reserve_interior_cells,
    sample_erase_mask,
)

# A normalized 24px square at 16..40px, covering cells [1:3, 1:3] of a
# 128px / 16px grid.
BOX_2X2 = torch.tensor([[0.21875, 0.21875, 24 / 128, 24 / 128]])


def make_image(size: int = 128, fill: float = 0.7) -> torch.Tensor:
    return torch.full((3, size, size), fill)


# ── protected_cell_mask ──────────────────────────────────────────────────


class TestProtectedCellMask:
    def test_overlapping_cells_protected(self):
        mask = protected_cell_mask(BOX_2X2, 128, 128, padding=0.0)
        assert mask.shape == (8, 8)
        assert mask[1:3, 1:3].all()
        assert not mask[0, :].any() and not mask[:, 0].any()
        # Default 0.15 dilation (2px floor) additionally protects the ring.
        dilated = protected_cell_mask(BOX_2X2, 128, 128)
        assert dilated.sum() > mask.sum()

    def test_margin_expands_protection(self):
        base = protected_cell_mask(BOX_2X2, 128, 128, padding=0.0)
        dilated = protected_cell_mask(BOX_2X2, 128, 128, padding=0.5)
        assert dilated.sum() > base.sum()
        # padding=0 disables dilation entirely (no 2px floor).
        assert torch.equal(base, protected_cell_mask(BOX_2X2, 128, 128, padding=0.0))

    def test_empty_boxes_protect_nothing(self):
        assert not protected_cell_mask(torch.zeros(0, 4), 128, 128).any()
        assert not protected_cell_mask(None, 128, 128).any()

    def test_tiny_box_protects_exactly_one_cell(self):
        # An 8px box fully inside one 16px cell (52..60 within 48..64):
        # any-overlap keeps it fully safe.
        tiny = torch.tensor([[0.4375, 0.4375, 8 / 128, 8 / 128]])
        mask = protected_cell_mask(tiny, 128, 128, padding=0.0)
        assert mask.sum() == 1


# ── sample_erase_mask ────────────────────────────────────────────────────


class TestSampleEraseMask:
    def test_exact_erase_count(self):
        protected = torch.zeros(8, 8, dtype=torch.bool)
        erase = sample_erase_mask(
            protected, 0.4, generator=torch.Generator().manual_seed(0)
        )
        assert int(erase.sum()) == int(round(0.4 * 64))

    def test_protected_cells_never_erased_under_repeated_sampling(self):
        boxes = torch.tensor([[0.25, 0.25, 0.3, 0.3], [0.7, 0.6, 0.25, 0.3]])
        protected = protected_cell_mask(boxes, 128, 128)
        generator = torch.Generator().manual_seed(1)
        for _ in range(20):
            erase = sample_erase_mask(protected, 0.9, generator=generator)
            assert not (erase & protected).any()

    def test_fully_protected_canvas_yields_empty_mask(self):
        protected = torch.ones(8, 8, dtype=torch.bool)
        assert not sample_erase_mask(protected, 0.4).any()


# ── erase_background ─────────────────────────────────────────────────────


class TestEraseBackground:
    def test_erased_cells_zero_and_protected_untouched(self):
        image = make_image()
        boxes = torch.tensor([[0.25, 0.25, 0.3, 0.3]])
        out = erase_background(
            image, boxes, 1.0, generator=torch.Generator().manual_seed(2)
        )
        protected = protected_cell_mask(boxes, 128, 128)
        cell = protected.repeat_interleave(16, 0).repeat_interleave(16, 1)
        assert torch.equal(out[:, cell], image[:, cell])  # protected pixels intact
        background = ~cell
        assert (out[:, background] == 0.0).all()  # everything else erased

    def test_single_cell_object_never_erased_even_at_full_ratio(self):
        tiny = torch.tensor([[0.4375, 0.4375, 8 / 128, 8 / 128]])
        image = make_image()
        out = erase_background(image, tiny, 1.0, padding=0.0)
        # The object's single cell survives at full ratio; the background goes.
        assert torch.equal(out[:, 48:64, 48:64], image[:, 48:64, 48:64])
        assert not torch.equal(out, image)

    def test_ratio_zero_and_missing_boxes_are_identity(self):
        image = make_image()
        assert torch.equal(erase_background(image, BOX_2X2, 0.0), image)
        assert torch.equal(erase_background(image, None, 0.5), image)
        assert torch.equal(erase_background(image, torch.zeros(0, 4), 0.5), image)

    def test_input_not_mutated(self):
        image = make_image()
        erase_background(
            image.clone(), BOX_2X2, 0.5, generator=torch.Generator().manual_seed(3)
        )
        assert torch.equal(image, make_image())

    def test_generator_reproducibility(self):
        out1 = erase_background(
            make_image(), BOX_2X2, 0.5, generator=torch.Generator().manual_seed(4)
        )
        out2 = erase_background(
            make_image(), BOX_2X2, 0.5, generator=torch.Generator().manual_seed(4)
        )
        assert torch.equal(out1, out2)


# ── interior mode (V2 constrained foreground erasure) ────────────────────


class TestInteriorMode:
    def test_min_keep_floor_respected(self):
        """Every box keeps at least ``min_keep`` interior cells across draws."""
        boxes = torch.tensor([[0.25, 0.25, 0.3, 0.3], [0.7, 0.6, 0.25, 0.3]])
        generator = torch.Generator().manual_seed(5)
        for _ in range(20):
            protected = protected_cell_mask(boxes, 128, 128)
            interior = protected_cell_mask(boxes, 128, 128, padding=0.0)
            reserved = reserve_interior_cells(
                boxes, 128, 128, cell=16, min_keep=2, generator=generator
            )
            effective = (protected & ~interior) | reserved
            erase = sample_erase_mask(effective, 1.0, generator=generator)
            for box in boxes.tolist():
                cx, cy, w, h = box
                x1, y1 = int((cx - w / 2) * 128 // 16), int((cy - h / 2) * 128 // 16)
                x2 = int(-(-(cx + w / 2) * 128 // 16))  # ceil: partial cells count
                y2 = int(-(-(cy + h / 2) * 128 // 16))
                kept = interior[y1:y2, x1:x2] & ~erase[y1:y2, x1:x2]
                assert int(kept.sum()) >= 2

    def test_tiny_boxes_fully_protected_even_with_interior(self):
        tiny = torch.tensor([[0.4375, 0.4375, 8 / 128, 8 / 128]])
        image = make_image()
        out = erase_background(image, tiny, 1.0, padding=0.0, interior=True, min_keep=2)
        # One-cell box reserves everything: its cell is untouched.
        assert torch.equal(out[:, 48:64, 48:64], image[:, 48:64, 48:64])

    def test_margin_ring_still_protected_with_interior_on(self):
        image = make_image()
        out = erase_background(
            image,
            BOX_2X2,
            1.0,
            padding=0.15,
            interior=True,
            min_keep=2,
            generator=torch.Generator().manual_seed(6),
        )
        ring = protected_cell_mask(BOX_2X2, 128, 128) & ~protected_cell_mask(
            BOX_2X2, 128, 128, padding=0.0
        )
        cell = ring.repeat_interleave(16, 0).repeat_interleave(16, 1)
        assert torch.equal(out[:, cell], image[:, cell])

    def test_background_still_erased_with_interior_on(self):
        out = erase_background(
            make_image(),
            BOX_2X2,
            0.5,
            interior=True,
            min_keep=2,
            generator=torch.Generator().manual_seed(7),
        )
        assert (out < make_image()).any()  # something got zero-filled


# ── transform integration ────────────────────────────────────────────────


def make_item(size: int = 128) -> dict:
    return {
        "image_path": "img.jpg",
        "image": torch.rand(3, size, size),
        "bboxes": tv_tensors.BoundingBoxes(
            # XYXY corners: 25.6px square centered at (51.2, 51.2).
            torch.tensor([[38.4, 38.4, 64.0, 64.0]]),
            format="XYXY",
            canvas_size=(size, size),
        ),
        "classes": torch.tensor([0]),
        "original_size": (100, 150),
    }


class TestTransformIntegration:
    def test_off_by_default(self):
        transform = RTDETRDetectionTransform(
            RTDETRDetectionTransformArgs(image_size=(128, 128))
        )
        item = make_item()
        out = transform(dict(item))
        assert out["image"].shape == (3, 128, 128)

    def test_erase_on_changes_background_and_keeps_boxes(self):
        args = RTDETRDetectionTransformArgs(
            image_size=(128, 128),
            photometric_distort_prob=None,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            horizontal_flip_prob=0.0,
            vertical_flip_prob=0.0,
            scale_jitter=None,
            mosaic_prob=None,
            random_erase_prob=1.0,
            random_erase_ratio=0.5,
            random_erase_interior=True,
            random_erase_min_keep=2,
        )
        transform = RTDETRDetectionTransform(args)
        item = make_item()
        out = transform(dict(item))
        assert out["image"].shape == (3, 128, 128)
        assert (out["image"] == 0.0).any()  # erasure applied
        # The GT box itself survives (a 25.6px box at 128px / 16px cells
        # spans 2x2 cells >= min_keep): at least one protected pixel keeps
        # its non-zero value inside the box region.
        box = out["bboxes"][0]
        x1, y1 = int((box[0] - box[2] / 2) * 128), int((box[1] - box[3] / 2) * 128)
        x2, y2 = int((box[0] + box[2] / 2) * 128), int((box[1] + box[3] / 2) * 128)
        assert (out["image"][:, y1:y2, x1:x2] > 0.0).any()

    def test_val_recipe_disables_erasure(self):
        args = RTDETRDetectionTransformArgs(
            image_size=(128, 128), random_erase_prob=0.5
        )
        assert build_val_transform_args(args).random_erase_prob is None


class TestEraseStepStop:
    """The Dome aug-stop cadence: erasure closes at ``random_erase_step_stop``."""

    def _args(self, step_stop) -> RTDETRDetectionTransformArgs:
        return RTDETRDetectionTransformArgs(
            image_size=(128, 128),
            photometric_distort_prob=None,
            random_zoom_out_prob=None,
            random_iou_crop_prob=None,
            horizontal_flip_prob=0.0,
            vertical_flip_prob=0.0,
            scale_jitter=None,
            mosaic_prob=None,
            random_erase_prob=1.0,
            random_erase_ratio=0.5,
            random_erase_step_stop=step_stop,
        )

    def test_active_before_stop(self):
        transform = RTDETRDetectionTransform(self._args(step_stop=100))
        transform.set_step(10)
        out = transform(dict(make_item()))
        assert (out["image"] == 0.0).any()

    def test_closed_after_stop(self):
        transform = RTDETRDetectionTransform(self._args(step_stop=100))
        transform.set_step(100)
        out = transform(dict(make_item()))
        assert not (out["image"] == 0.0).any()

    def test_none_is_constant(self):
        transform = RTDETRDetectionTransform(self._args(step_stop=None))
        transform.set_step(10_000)
        out = transform(dict(make_item()))
        assert (out["image"] == 0.0).any()

    def test_auto_resolves_in_resolve_step_schedule(self):
        args = self._args(step_stop="auto")
        args.resolve_step_schedule(
            total_steps=1000, train_num_batches=100, gradient_accumulation_steps=1
        )
        assert args.random_erase_step_stop == 800

    def test_stop_boundary_triggers_dataloader_reinit(self):
        """The erase window rides the step activation tuple: crossing the
        stop flips the gate AND marks the dataloader for reinitialization —
        the mechanism that carries the new gate into (persistent) worker
        processes. A call-time step read would see each worker's stale
        pickled copy for any window that does not coincide with another
        op's boundary."""
        transform = RTDETRDetectionTransform(self._args(step_stop=100))
        transform.set_step(99)
        assert transform.compose.erase_active()
        assert not transform.requires_dataloader_reinitialization()
        transform.set_step(100)
        assert not transform.compose.erase_active()
        assert transform.requires_dataloader_reinitialization()
        transform.mark_dataloader_as_reinitialized()
        assert not transform.requires_dataloader_reinitialization()

    def test_gate_inactive_when_prob_unset(self):
        args = self._args(step_stop=100)
        args.random_erase_prob = None
        transform = RTDETRDetectionTransform(args)
        transform.set_step(10)
        assert not transform.compose.erase_active()
