# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""RF-DETR model cards: official metrics tied to pinned HF bytes.

The cards are the Phase 0 answer to "what license / which bytes / what
AP" for the five registered RF-DETR checkpoints. They record the Roboflow
official COCO numbers (not our reproductions) and must stay in lockstep
with ``WEIGHT_MANIFEST``.
"""

from __future__ import annotations

from autovision.encoders.weight_hashes import WEIGHT_MANIFEST
from autovision.registry.model_cards import rfdetr_model_cards

CARDS = rfdetr_model_cards(WEIGHT_MANIFEST)

_ACTIVE = {
    "rfdetr/rf-detr-nano": (48.4, 67.6, 30.5, 384.0),
    "rfdetr/rf-detr-small": (53.0, 72.1, 32.1, 512.0),
    "rfdetr/rf-detr-medium": (54.7, 73.6, 33.7, 576.0),
    "rfdetr/rf-detr-large": (56.5, 75.1, 33.9, 704.0),
}
_LEGACY_BASE = "rfdetr/rf-detr-base"


class TestRFDetrCards:
    def test_five_registered_checkpoints_have_cards(self):
        expected = {
            repo_id
            for repo_id in WEIGHT_MANIFEST
            if repo_id.startswith("Roboflow/rf-detr-")
        }
        assert len(expected) == 5
        assert {card.provenance["repo_id"] for card in CARDS.values()} == expected

    def test_cards_are_valid_and_apache_licensed(self):
        for key, card in CARDS.items():
            card.validate()
            assert card.license == "Apache-2.0", key
            assert card.seed is None, key
            assert set(card.weights_sha256) == {"model.safetensors"}, key
            assert len(next(iter(card.weights_sha256.values()))) == 64, key

    def test_cards_pin_the_manifest_sha256_and_revision(self):
        for card in CARDS.values():
            artifact = WEIGHT_MANIFEST[card.provenance["repo_id"]]
            assert artifact.license == "apache-2.0"
            assert card.weights_sha256["model.safetensors"] == artifact.sha256[
                "model.safetensors"
            ]
            assert card.provenance["revision"] == artifact.revision
            assert card.version == artifact.revision[:8]

    def test_active_tiers_carry_official_coco_metrics(self):
        for key, (ap, ap50, params, input_px) in _ACTIVE.items():
            card = CARDS[key]
            assert card.metrics["coco_map"] == ap
            assert card.metrics["coco_map50"] == ap50
            assert card.metrics["params_m"] == params
            assert card.metrics["input_px"] == input_px
            assert card.metrics["coco_map"] > 0

    def test_base_is_legacy_and_not_advertised(self):
        card = CARDS[_LEGACY_BASE]
        assert card.metrics == {}
        assert "弃用" in card.known_limitations
