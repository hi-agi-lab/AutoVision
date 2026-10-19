# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Acceptance tests for the clean-provenance P0 rewrites (design doc 28).

Each rewritten unit is checked against an *independently constructed*
oracle (naive loops / closed-form math), not against the previous
implementation. Specs: the clean-room specs S1-S4 (private provenance copy).
"""

import math
import random

import torch

from autovision.pretext.dinov3 import DINOv3
from autovision.pretext.dinov3.constrained_crop import sample_high_overlap_box
from autovision.pretext.dinov3.dinov3 import _restore_orientation
from autovision.pretext.dinov3.paika import _roi_resample_to_grid

# ── S-1: sample_high_overlap_box ────────────────────────────────────────


class TestSampleBox:
    def test_containment_and_area_band_both_parent_aspects(self):
        # Wide (aspect 0.5) and tall (aspect 2.0) parents: the width/height
        # clamping path must keep every draw inside the acceptance band.
        parents = [(10, 20, 110, 220), (0, 0, 200, 100)]
        for parent in parents:
            rng = random.Random(1234)
            parent_area = (parent[2] - parent[0]) * (parent[3] - parent[1])
            for _ in range(1000):
                x0, y0, x1, y1 = sample_high_overlap_box(parent, 0.9, rng)
                assert parent[0] <= x0 < x1 <= parent[2]
                assert parent[1] <= y0 < y1 <= parent[3]
                ratio = (x1 - x0) * (y1 - y0) / parent_area
                assert 0.04 < ratio <= 0.41, f"{parent}: draw {ratio:.3f}"

    def test_degenerate_parent_returned_as_is(self):
        assert sample_high_overlap_box((5, 5, 6, 6), 0.9, random.Random(0)) == (5, 5, 6, 6)

    def test_deterministic_given_rng_only(self):
        boxes_a = [sample_high_overlap_box((0, 0, 50, 50), 0.9, random.Random(7)) for _ in range(20)]
        boxes_b = [sample_high_overlap_box((0, 0, 50, 50), 0.9, random.Random(7)) for _ in range(20)]
        assert boxes_a == boxes_b


# ── S-2: _roi_align_view + resample oracle ──────────────────────────────


def naive_resample(feat: torch.Tensor, box, out_h: int, out_w: int) -> torch.Tensor:
    """Slow per-cell bilinear oracle (corner coords, center sampling,
    zero padding) — an independent construction of the grid_sample path."""
    channels, height, width = feat.shape
    x0, y0, x1, y1 = (float(value) for value in box)
    padded = torch.zeros(channels, height + 2, width + 2, dtype=torch.float64)
    padded[:, 1:-1, 1:-1] = feat.double()
    out = torch.zeros(channels, out_h, out_w, dtype=torch.float64)
    for row in range(out_h):
        for col in range(out_w):
            # Cell-center sampling in corner coordinates…
            x = x0 + (col + 0.5) / out_w * (x1 - x0)
            y = y0 + (row + 0.5) / out_h * (y1 - y0)
            # …which grid_sample(align_corners=False) reads at pixel-center
            # coordinate x - 0.5; shift by +1 into the zero-padded array.
            px = x + 0.5
            py = y + 0.5
            xl, yl = math.floor(px), math.floor(py)
            fx, fy = px - xl, py - yl
            for c in range(channels):
                top = padded[c, yl, xl] * (1 - fx) + padded[c, yl, xl + 1] * fx
                bottom = padded[c, yl + 1, xl] * (1 - fx) + padded[c, yl + 1, xl + 1] * fx
                out[c, row, col] = top * (1 - fy) + bottom * fy
    return out


class TestRoiResampleOracle:
    def test_matches_naive_bilinear_oracle(self):
        torch.manual_seed(0)
        feat = torch.randn(1, 3, 6, 5, dtype=torch.float64)
        boxes = torch.tensor(
            [
                [0.0, 0.0, 5.0, 6.0],  # identity
                [1.5, 0.75, 4.25, 5.5],  # sub-box, non-divisible grid
                [0.0, 0.0, 2.5, 3.0],  # quarter map
            ],
            dtype=torch.float64,
        )
        for box in boxes:
            got = _roi_resample_to_grid(feat, box[None], out_h=4, out_w=7)
            want = naive_resample(feat[0], box, out_h=4, out_w=7)
            assert torch.allclose(got[0], want, atol=1e-8), box


class TestRoiAlignView:
    @staticmethod
    def call(tokens, geom, region, in_h, in_w, out_h, out_w):
        return DINOv3._roi_align_view(
            tokens, geom, region, in_h=in_h, in_w=in_w, out_h=out_h, out_w=out_w
        )

    def test_matches_oracle_with_flips(self):
        torch.manual_seed(1)
        batch, channels, grid = 3, 2, 4
        tokens = torch.randn(batch, channels, grid, grid, dtype=torch.float64)
        # Mixed flip flags across the batch; crops cover distinct sub-boxes
        # of a 100x100 image.
        geom = torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 0.0, 0.0],
                [10.0, 20.0, 90.0, 100.0, 100.0, 100.0, 1.0, 0.0],
                [0.0, 0.0, 50.0, 60.0, 100.0, 100.0, 1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        region = torch.tensor(
            [
                [25.0, 25.0, 75.0, 75.0],
                [20.0, 30.0, 90.0, 100.0],
                [10.0, 15.0, 50.0, 60.0],
            ],
            dtype=torch.float64,
        )
        got = self.call(tokens, geom, region, in_h=grid, in_w=grid, out_h=3, out_w=5)
        for b in range(batch):
            x0, y0, x1, y1 = region[b]
            crop_x0, crop_y0, crop_x1, crop_y1 = geom[b, :4]
            gx0 = (x0 - crop_x0) / (crop_x1 - crop_x0) * grid
            gx1 = (x1 - crop_x0) / (crop_x1 - crop_x0) * grid
            gy0 = (y0 - crop_y0) / (crop_y1 - crop_y0) * grid
            gy1 = (y1 - crop_y0) / (crop_y1 - crop_y0) * grid
            unflipped = tokens[b]
            if geom[b, 6] > 0.5:
                unflipped = unflipped.flip(-1)
            if geom[b, 7] > 0.5:
                unflipped = unflipped.flip(-2)
            want = naive_resample(
                unflipped, (gx0, gy0, gx1, gy1), out_h=3, out_w=5
            )
            assert torch.allclose(got[b], want, atol=1e-8), b

    def test_restore_orientation_groups(self):
        tokens = torch.arange(2 * 1 * 2 * 2, dtype=torch.float64).reshape(2, 1, 2, 2)
        hflip = torch.tensor([True, False])
        vflip = torch.tensor([False, True])
        out = _restore_orientation(tokens, hflip, vflip)
        assert torch.equal(out[0], tokens[0].flip(-1))
        assert torch.equal(out[1], tokens[1].flip(-2))

    def test_restore_orientation_keeps_grad(self):
        tokens = torch.randn(2, 1, 2, 2, requires_grad=True)
        out = _restore_orientation(
            tokens, torch.tensor([True, True]), torch.tensor([False, True])
        )
        out.sum().backward()
        assert tokens.grad is not None
        assert torch.isfinite(tokens.grad).all()


# ── S-3: _align_cross_view_pair ─────────────────────────────────────────


def geom_row(x0, y0, x1, y1, hflip=0.0, vflip=0.0):
    return torch.tensor([[x0, y0, x1, y1, 100.0, 100.0, hflip, vflip]])


class TestAlignCrossViewPair:
    @staticmethod
    def run(student_geom, teacher_geom, s_grid=4, t_grid=6, out=4):
        tokens_s = torch.randn(1, 3, s_grid, s_grid, requires_grad=True)
        tokens_t = torch.randn(1, 3, t_grid, t_grid)
        method = object.__new__(DINOv3)  # the pair aligner is self-free
        student_al, teacher_al, valid = method._align_cross_view_pair(
            student_tokens=tokens_s,
            student_geom=student_geom,
            student_h=s_grid,
            student_w=s_grid,
            teacher_tokens=tokens_t,
            teacher_geom=teacher_geom,
            teacher_h=t_grid,
            teacher_w=t_grid,
            out_h=out,
            out_w=out,
        )
        return tokens_s, tokens_t, student_al, teacher_al, valid

    def test_contained_local_overlaps_fully(self):
        s, t, s_al, t_al, valid = self.run(
            geom_row(10, 10, 40, 40), geom_row(0, 0, 100, 100)
        )
        assert s_al.shape == (1, 3, 4, 4)
        assert t_al.shape == (1, 3, 4, 4)
        assert valid.all()
        assert s_al.requires_grad and not t_al.requires_grad

    def test_disjoint_crops_marked_invalid(self):
        _, _, _, _, valid = self.run(
            geom_row(0, 0, 10, 10), geom_row(50, 50, 100, 100)
        )
        assert not valid.any()

    def test_sliver_overlap_marked_invalid(self):
        # Intersection height 0.5 px < one output cell (100/4 = 25 px).
        _, _, _, _, valid = self.run(
            geom_row(0, 0, 100, 4), geom_row(0, 3.5, 100, 100)
        )
        assert not valid.any()

    def test_gradients_reach_student_tokens_only(self):
        s, t, s_al, t_al, valid = self.run(
            geom_row(10, 10, 40, 40, hflip=1.0), geom_row(0, 0, 100, 100)
        )
        spatial_valid = valid.view(1, 1, 4, 4)
        (s_al * spatial_valid).sum().backward()
        assert s.grad is not None
        assert float(s.grad.abs().sum()) > 0.0
        assert not t.requires_grad
