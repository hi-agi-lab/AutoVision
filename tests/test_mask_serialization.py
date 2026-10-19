# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Mask serialization at the serving and CLI boundaries."""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from autovision.cli.model_handle import predict_to_files
from autovision.errors import AutoVisionError

pytest.importorskip("fastapi")


def _b64_image(size: int = 16, seed: int = 0) -> str:
    rng = np.random.default_rng(seed)
    array = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _record() -> dict:
    masks = torch.zeros(1, 16, 16, dtype=torch.bool)
    masks[0, 4:12, 4:12] = True
    return {
        "bboxes": torch.tensor([[4, 4, 12, 12]], dtype=torch.float32),
        "scores": torch.tensor([0.9]),
        "labels": torch.tensor([0]),
        "masks": masks,
    }


class _MaskEngine:
    backend = "torch"

    def run_batch(self, images, threshold=0.5):
        return [_record() for _ in images]


class _MaskNetwork:
    def predict(self, image, **kwargs):
        return _record()


class _ShortEngine(_MaskEngine):
    def run_batch(self, images, threshold=0.5):
        return []


class _NanEngine(_MaskEngine):
    def run_batch(self, images, threshold=0.5):
        records = super().run_batch(images, threshold=threshold)
        for record in records:
            record["scores"] = torch.tensor([float("nan")])
        return records


class _ObbNetwork:
    def predict(self, image, **kwargs):
        return {
            "polygons": torch.tensor([[[2, 2], [14, 2], [14, 14], [2, 14]]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
        }


class TestServingMasks:
    def test_rle_is_the_default_and_is_lossless(self):
        from fastapi.testclient import TestClient
        from pycocotools import mask as mask_utils

        from autovision.serving import create_app

        client = TestClient(create_app(engine=_MaskEngine(), max_batch=1))
        payload = client.post("/predict", json={"image": _b64_image()}).json()
        encoded = payload["results"][0]["masks"][0]
        assert encoded["size"] == [16, 16]
        decoded = mask_utils.decode(
            {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}
        )
        expected = np.zeros((16, 16), dtype=np.uint8)
        expected[4:12, 4:12] = 1
        assert np.array_equal(decoded.astype(np.uint8), expected)

    def test_mask_format_none_omits_masks(self):
        from fastapi.testclient import TestClient

        from autovision.serving import create_app

        client = TestClient(
            create_app(engine=_MaskEngine(), max_batch=1, mask_format="none")
        )
        payload = client.post("/predict", json={"image": _b64_image()}).json()
        assert "masks" not in payload["results"][0]

    def test_invalid_mask_format_is_rejected(self):
        from autovision.serving import create_app

        with pytest.raises(AutoVisionError, match="mask_format"):
            create_app(engine=_MaskEngine(), mask_format="png")

    def test_short_engine_result_is_a_500(self):
        from fastapi.testclient import TestClient

        from autovision.serving import create_app

        client = TestClient(
            create_app(engine=_ShortEngine(), max_batch=1),
            raise_server_exceptions=False,
        )
        response = client.post("/predict", json={"image": _b64_image()})
        assert response.status_code == 500

    def test_non_finite_scores_are_json_null(self):
        from fastapi.testclient import TestClient

        from autovision.serving import create_app

        client = TestClient(create_app(engine=_NanEngine(), max_batch=1))
        payload = client.post("/predict", json={"image": _b64_image()}).json()
        assert payload["results"][0]["scores"] == [None]


class TestPredictToFiles:
    def test_instance_masks_are_written_as_rle(self, tmp_path):
        image = tmp_path / "img.png"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image)
        out = predict_to_files(
            _MaskNetwork(), image, tmp_path / "out", save_visuals=False
        )
        record = json.loads((out / "predictions.json").read_text())[0]
        assert record["masks"][0]["size"] == [16, 16]
        assert "counts" in record["masks"][0]

    def test_visuals_work_for_oriented_boxes(self, tmp_path):
        image = tmp_path / "img.png"
        Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)).save(image)
        out = predict_to_files(
            _ObbNetwork(), image, tmp_path / "out", save_visuals=True
        )
        assert (out / "visuals" / "img_pred.png").exists()

    def test_mask_format_validation(self, tmp_path):
        with pytest.raises(AutoVisionError, match="mask_format"):
            predict_to_files(
                _MaskNetwork(),
                tmp_path / "x.png",
                tmp_path / "out",
                mask_format="png",
            )
