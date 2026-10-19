# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Release gate: a ``released`` spectrum tier must be a shippable product.

The gate is intentionally stricter than ``ModelCard.validate``: it also
requires a full-COCO AP reading, a latency record, and a reproduction
script that actually exists. Phase 0 has no released tier yet, so the
real registry must pass vacuously; the negative cases below lock the gate
before Phase 1 flips n/s to released.
"""

from __future__ import annotations

from autovision.registry.model_card import LatencyRecord, ModelCard
from autovision.registry.spectrum_release import (
    ReleasedSpectrumWeight,
    validate_spectrum_release,
)
from autovision.tasks.detection.detr.variants import SPECTRUM, SpectrumTier

_SHA = "a" * 64


def _tier(
    *, tier: str = "n", status: str = "released", variant_key: str = "hgnetv2-b0"
) -> dict[str, SpectrumTier]:
    return {
        tier: SpectrumTier(
            tier=tier,
            variant_key=variant_key,
            route="distill",
            benchmark_anchor="test anchor",
            status=status,
        )
    }


def _latency(script: str = "scripts/measure_spectrum_benchmarks.py") -> LatencyRecord:
    return LatencyRecord(
        backend="tensorrt",
        precision="fp16",
        hardware="NVIDIA T4",
        batch_size=1,
        p50_ms=2.0,
        script=script,
    )


def _card(**overrides) -> ModelCard:
    fields = {
        "model": "detr/n",
        "name": "DETR-N (test)",
        "train_dataset": "COCO 2017 train2017",
        "schedule": "12ep distill",
        "seed": 0,
        "metrics": {"coco_map": 40.0},
        "latency": [_latency()],
        "license": "Apache-2.0",
        "weights_sha256": {"detr_n.pt": _SHA},
    }
    fields.update(overrides)
    return ModelCard(**fields)


def _asset(**overrides) -> ReleasedSpectrumWeight:
    fields = {
        "tier": "n",
        "asset": "hf://test/detr-n",
        "weights_file": "detr_n.pt",
        "card": _card(),
    }
    fields.update(overrides)
    return ReleasedSpectrumWeight(**fields)


class TestRealRegistry:
    def test_no_tier_is_released_yet(self):
        # n/s are spec-frozen; m/l/x are planned. If this test starts
        # failing, Phase 1 published weights — add the asset + card first.
        assert SPECTRUM["n"].status == "spec-frozen"
        assert SPECTRUM["s"].status == "spec-frozen"
        assert {entry.status for entry in SPECTRUM.values()} == {
            "spec-frozen",
            "planned",
        }

    def test_real_registry_passes_the_gate(self):
        assert validate_spectrum_release(SPECTRUM) == []


class TestReleasedGate:
    def test_valid_release_passes(self):
        assert validate_spectrum_release(_tier(), {"n": _asset()}) == []

    def test_release_without_asset_fails(self):
        problems = validate_spectrum_release(_tier(), {})
        assert any("no published weight asset" in problem for problem in problems)

    def test_release_without_model_key_fails(self):
        problems = validate_spectrum_release(
            _tier(variant_key=""), {"n": _asset()}
        )
        assert any("no model key" in problem for problem in problems)

    def test_card_model_must_match_tier(self):
        problems = validate_spectrum_release(
            _tier(), {"n": _asset(card=_card(model="detr/s"))}
        )
        assert any("expected 'detr/n'" in problem for problem in problems)

    def test_weights_file_must_be_pinned(self):
        problems = validate_spectrum_release(
            _tier(), {"n": _asset(weights_file="other.pt")}
        )
        assert any("not pinned" in problem for problem in problems)

    def test_full_coco_ap_is_required(self):
        problems = validate_spectrum_release(
            _tier(), {"n": _asset(card=_card(metrics={}))}
        )
        assert any("no full-COCO AP" in problem for problem in problems)

    def test_seed_is_required_for_our_own_weights(self):
        problems = validate_spectrum_release(
            _tier(), {"n": _asset(card=_card(seed=None))}
        )
        assert any("training seed is required" in problem for problem in problems)

    def test_latency_is_required(self):
        problems = validate_spectrum_release(
            _tier(), {"n": _asset(card=_card(latency=[]))}
        )
        assert any("no latency record" in problem for problem in problems)

    def test_latency_script_must_exist(self):
        card = _card(latency=[_latency("scripts/does_not_exist.py")])
        problems = validate_spectrum_release(_tier(), {"n": _asset(card=card)})
        assert any("does not exist" in problem for problem in problems)

    def test_non_released_tier_cannot_claim_an_asset(self):
        problems = validate_spectrum_release(
            _tier(status="spec-frozen"), {"n": _asset()}
        )
        assert any("has a published weight asset" in problem for problem in problems)

    def test_spec_frozen_requires_a_model_key(self):
        problems = validate_spectrum_release(
            _tier(status="spec-frozen", variant_key=""), {}
        )
        assert any("no model key" in problem for problem in problems)

    def test_asset_for_unknown_tier_fails(self):
        problems = validate_spectrum_release(
            _tier(), {"n": _asset(), "z": _asset(tier="z", card=_card(model="detr/z"))}
        )
        assert any("has no spectrum tier" in problem for problem in problems)
