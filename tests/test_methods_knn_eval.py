# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the in-training kNN probe (design §7.5)."""

from pathlib import Path

import pytest
import torch
from torch import nn

from autovision.augment.transform_args import (
    MethodTransformArgs,
    NormalizeArgs,
    RandomResizeArgs,
)
from autovision.pretext.knn_eval import (
    KnnEvalArgs,
    KnnEvalRunner,
    attach_knn_eval_runner,
    knn_accuracy,
    labels_for_filenames,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_transform_args(image_size: tuple[int, int] = (16, 16)) -> MethodTransformArgs:
    return MethodTransformArgs(
        image_size=image_size,
        random_resize=RandomResizeArgs(min_scale=1.0, max_scale=1.0),
        normalize=NormalizeArgs(),
    )


class TestLabelsForFilenames:
    def test_imagefolder_layout_gives_labels(self, tmp_path: Path):
        names = [
            str(tmp_path / "cat" / "a.jpg"),
            str(tmp_path / "cat" / "b.jpg"),
            str(tmp_path / "dog" / "c.jpg"),
        ]
        assert labels_for_filenames(names) == ["cat", "cat", "dog"]

    def test_flat_layout_has_no_labels(self, tmp_path: Path):
        names = [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")]
        assert labels_for_filenames(names) is None

    def test_single_class_has_no_labels(self, tmp_path: Path):
        names = [str(tmp_path / "cat" / "a.jpg"), str(tmp_path / "cat" / "b.jpg")]
        assert labels_for_filenames(names) is None

    def test_mmap_dict_entries_supported(self, tmp_path: Path):
        names = [
            {"filenames": str(tmp_path / "cat" / "a.jpg")},
            {"filenames": str(tmp_path / "dog" / "b.jpg")},
        ]
        assert labels_for_filenames(names) == ["cat", "dog"]


class TestKnnEvalArgs:
    def test_auto_resolves_small_dataset_on(self):
        args = KnnEvalArgs(enabled="auto")
        args.resolve_auto_fields(dataset_size=100)
        assert args.enabled is True
        assert args.was_auto is True

    def test_auto_resolves_large_dataset_off(self):
        args = KnnEvalArgs(enabled="auto")
        args.resolve_auto_fields(dataset_size=500_001)
        assert args.enabled is False

    def test_auto_requires_dataset_size(self):
        args = KnnEvalArgs(enabled="auto")
        with pytest.raises(ValueError, match="dataset size"):
            args.resolve_auto_fields(dataset_size=None)


class TestKnnAccuracy:
    def test_separable_clusters_score_one(self):
        generator = torch.Generator().manual_seed(0)
        # Two well-separated clusters, 32 samples each: with a ~50% gallery
        # split every query keeps enough same-class neighbors for a majority
        # vote at k=5.
        class_a = torch.nn.functional.normalize(
            torch.randn(32, 16, generator=generator) * 0.1 + 1.0, dim=-1
        )
        class_b = torch.nn.functional.normalize(
            torch.randn(32, 16, generator=generator) * 0.1 - 1.0, dim=-1
        )
        embeddings = torch.cat([class_a, class_b])
        labels = torch.tensor([0] * 32 + [1] * 32)
        result = knn_accuracy(embeddings, labels, k=5, split=0.5)
        assert result["accuracy_top1"] == pytest.approx(1.0)
        assert result["accuracy_top5"] == pytest.approx(1.0)

    def test_no_top5_key_when_k_below_five(self):
        generator = torch.Generator().manual_seed(0)
        embeddings = torch.nn.functional.normalize(
            torch.randn(8, 16, generator=generator), dim=-1
        )
        labels = torch.tensor([0, 1] * 4)
        result = knn_accuracy(embeddings, labels, k=3, split=0.5)
        assert "accuracy_top5" not in result

    def test_k_clamped_to_gallery_size(self):
        embeddings = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
        labels = torch.tensor([0, 1, 0, 1])
        result = knn_accuracy(embeddings, labels, k=100, split=0.5)
        assert 0.0 <= result["accuracy_top1"] <= 1.0

    def test_degenerate_split_returns_nan(self):
        import math

        embeddings = torch.nn.functional.normalize(torch.randn(1, 8), dim=-1)
        labels = torch.tensor([0])
        result = knn_accuracy(embeddings, labels, k=1, split=0.5)
        assert math.isnan(result["accuracy_top1"])


class _ClusterEmbedding(nn.Module):
    """Embedding model that maps class-subdir images to separable clusters.

    Uses the image mean as a deterministic seed so two synthetic classes land
    in distinguishable regions regardless of the actual pixel content.
    """

    def __init__(self, num_channels: int = 3) -> None:
        super().__init__()
        self.embed_dim = 8
        self.proj = nn.Conv2d(num_channels, 8, kernel_size=1)

    def forward(self, x, pool: bool = False):
        features = self.proj(x)
        pooled = features.mean(dim=(2, 3), keepdim=True)
        if pool:
            return pooled
        return features

    def feature_maps(self, x):
        return {"features": self.proj(x)}

    def pool_features(self, x):
        return {"pooled_features": x["features"].mean(dim=(2, 3), keepdim=True)}

    def encoder_dim(self):
        return 8

    def raw_module(self):
        return self


class TestKnnEvalRunner:
    def _make_runner(self, tmp_path: Path, **knn_kwargs) -> tuple[KnnEvalRunner, nn.Module]:
        from tests.conftest import create_images

        data_dir = tmp_path / "data"
        create_images(data_dir / "class_a", num_images=4, size=(16, 16))
        create_images(data_dir / "class_b", num_images=4, size=(16, 16))
        filenames = sorted(str(p) for p in data_dir.rglob("*.jpg"))
        embedding_model = _ClusterEmbedding()
        embedding_model.eval()
        method = nn.Module()
        method.embedding_model = embedding_model  # type: ignore[attr-defined]
        args = KnnEvalArgs(enabled=True, k=2, split=0.5, batch_size=4, **knn_kwargs)
        runner = KnnEvalRunner(
            args=args,
            method=method,
            filenames=filenames,
            transform_args=_make_transform_args(),
        )
        return runner, method

    def test_run_returns_metrics(self, tmp_path: Path):
        runner, _ = self._make_runner(tmp_path)
        metrics = runner.run()
        assert "knn_eval/accuracy_top1" in metrics
        assert 0.0 <= metrics["knn_eval/accuracy_top1"] <= 1.0

    def test_should_run_respects_every_n_epochs(self, tmp_path: Path):
        runner, _ = self._make_runner(tmp_path, every_n_epochs=3)
        assert not runner.should_run(0)
        assert not runner.should_run(1)
        assert runner.should_run(2)

    def test_run_restores_train_mode(self, tmp_path: Path):
        runner, method = self._make_runner(tmp_path)
        method.embedding_model.train()
        runner.run()
        assert method.embedding_model.training

    def test_max_samples_downsamples(self, tmp_path: Path):
        runner, _ = self._make_runner(tmp_path, max_samples=4)
        dataset = runner._probe_dataset()
        assert len(dataset) == 4


class TestAttachKnnEvalRunner:
    def _make_method_args(self, **method_args_overrides):
        from autovision.pretext.distillationv3 import DistillationV3Args

        overrides = {"queue_size": 16}
        overrides.update(method_args_overrides)
        args = DistillationV3Args(**overrides)
        args.resolve_auto_fields(scaling=None)
        return args

    def test_attached_when_enabled(self, tmp_path: Path):
        from tests.conftest import create_images

        create_images(tmp_path, num_images=4, size=(8, 8))
        filenames = sorted(str(p) for p in tmp_path.rglob("*.jpg"))
        method = nn.Module()
        args = self._make_method_args(knn_eval={"enabled": True})
        runner = attach_knn_eval_runner(
            method=method,
            method_args=args,
            filenames=filenames,
            transform_args=_make_transform_args(),
        )
        assert runner is not None
        assert method.knn_eval_runner is runner  # type: ignore[attr-defined]

    def test_explicit_enabled_without_labels_raises(self, tmp_path: Path):
        flat = tmp_path / "flat"
        flat.mkdir()
        from tests.conftest import create_image

        filenames = [str(create_image(flat / f"img_{i}.jpg", seed=i)) for i in range(4)]
        method = nn.Module()
        args = self._make_method_args(knn_eval={"enabled": True})
        with pytest.raises(ValueError, match="ImageFolder"):
            attach_knn_eval_runner(
                method=method,
                method_args=args,
                filenames=filenames,
                transform_args=_make_transform_args(),
            )

    def test_auto_downgrades_to_disabled_without_labels(self, tmp_path: Path):
        flat = tmp_path / "flat"
        flat.mkdir()
        from tests.conftest import create_image

        filenames = [str(create_image(flat / f"img_{i}.jpg", seed=i)) for i in range(4)]
        method = nn.Module()
        args = self._make_method_args(knn_eval={"enabled": "auto"})
        args.knn_eval.resolve_auto_fields(dataset_size=10)  # auto -> enabled=True
        assert args.knn_eval.enabled is True
        runner = attach_knn_eval_runner(
            method=method,
            method_args=args,
            filenames=filenames,
            transform_args=_make_transform_args(),
        )
        assert runner is None

    def test_disabled_returns_none(self, tmp_path: Path):
        method = nn.Module()
        args = self._make_method_args(knn_eval={"enabled": False})
        runner = attach_knn_eval_runner(
            method=method,
            method_args=args,
            filenames=[],
            transform_args=_make_transform_args(),
        )
        assert runner is None


class TestPretrainKnnIntegration:
    def test_pretrain_logs_knn_metrics(self, tmp_path: Path):
        from tests.conftest import create_images

        from autovision import pretrain

        image_dir = tmp_path / "data"
        create_images(image_dir, num_images=8, size=(32, 32))
        result = pretrain(
            out=tmp_path / "run_knn",
            data=image_dir,
            model="torchvision/resnet18",
            epochs=2,
            batch_size=4,
            num_workers=0,
            method_args={"queue_size": 4, "knn_eval": {"enabled": True, "k": 2}},
        )
        assert "knn_eval/accuracy_top1" in result.metrics
        assert 0.0 <= result.metrics["knn_eval/accuracy_top1"] <= 1.0
        # Logged to the metrics stream too (two epochs -> two probe runs).
        lines = (result.out_dir / "metrics.jsonl").read_text().strip().splitlines()
        knn_lines = [line for line in lines if "knn_eval" in line]
        assert len(knn_lines) == 2
