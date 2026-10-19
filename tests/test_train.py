# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""End-to-end smoke tests: pretrain -> artifacts -> resume -> export/load."""


import pytest
import torch

from autovision import list_methods, list_models, pretrain
from autovision.cli.train import PretrainConfig

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    from tests.conftest import create_images

    directory = tmp_path_factory.mktemp("data") / "images"
    create_images(directory, num_images=8, size=(48, 48))
    return directory


def make_config(out, image_dir, **overrides) -> PretrainConfig:
    defaults = dict(
        out=out,
        data=image_dir,
        model="torchvision/resnet18",
        epochs=1,
        batch_size=4,
        num_workers=0,
        method_args={"queue_size": 4},
    )
    defaults.update(overrides)
    return PretrainConfig(**defaults)


class TestPretrainEndToEnd:
    def test_full_run_produces_all_artifacts(self, tmp_path, image_dir):
        result = pretrain(
            out=tmp_path / "run1",
            data=image_dir,
            model="torchvision/resnet18",
            epochs=1,
            batch_size=4,
            num_workers=0,
            method_args={"queue_size": 4},
        )
        out = result.out_dir
        # Artifacts: checkpoint, export, metrics, mmap-independent out dir.
        assert result.checkpoint_path.exists()
        assert result.exported_model_path.exists()
        assert (out / "train.log").exists()
        assert (out / "metrics.jsonl").exists()
        assert (out / "checkpoints" / "last.ckpt").exists()
        lines = (out / "metrics.jsonl").read_text().strip().splitlines()
        assert any("hyperparams" in line for line in lines)
        assert any("train_loss" in line for line in lines)

    def test_exported_model_loads(self, tmp_path, image_dir):
        result = pretrain(
            out=tmp_path / "run2",
            data=image_dir,
            model="torchvision/resnet18",
            epochs=1,
            batch_size=4,
            num_workers=0,
            method_args={"queue_size": 4},
        )
        from autovision import load_embedding_model

        exported = torch.load(result.exported_model_path, weights_only=True)
        assert exported["schema_version"] == 1
        assert exported["kind"] == "embedding_model"
        assert exported["model_name"] == "torchvision/resnet18"
        assert exported["license_info"]
        embedding_model = load_embedding_model(
            result.exported_model_path, device="cpu"
        )
        assert not embedding_model.training
        with torch.no_grad():
            features = embedding_model(torch.randn(1, 3, 224, 224), pool=True)
        assert features.shape[0] == 1

    def test_resume_interrupted_continues(self, tmp_path, image_dir):
        out = tmp_path / "run3"
        pretrain(
            out=out, data=image_dir, model="torchvision/resnet18",
            epochs=1, batch_size=4, num_workers=0, method_args={"queue_size": 4},
        )
        first_ckpt = (out / "checkpoints" / "last.ckpt").read_bytes()
        # Resume with a larger budget: continues from epoch 1 (no restart).
        result = pretrain(
            out=out, data=image_dir, model="torchvision/resnet18",
            epochs=2, batch_size=4, num_workers=0, method_args={"queue_size": 4},
            resume_interrupted=True,
        )
        assert result.epochs_run == 1  # only the remaining epoch ran
        second_ckpt = (out / "checkpoints" / "last.ckpt").read_bytes()
        assert first_ckpt != second_ckpt  # checkpoint updated
        from autovision.runtime.checkpoint import Checkpoint

        metadata = Checkpoint.from_path(out / "checkpoints" / "last.ckpt").autovision
        assert metadata.epoch == 1  # completed epoch 1 (0-indexed)
        assert metadata.global_step == 4  # 2 steps/epoch x 2 epochs

    def test_weights_only_checkpoint(self, tmp_path, image_dir):
        first = pretrain(
            out=tmp_path / "run4a", data=image_dir, model="torchvision/resnet18",
            epochs=1, batch_size=4, num_workers=0, method_args={"queue_size": 4},
        )
        # Initialize a new run from the first run's weights.
        pretrain(
            out=tmp_path / "run4b", data=image_dir, model="torchvision/resnet18",
            epochs=1, batch_size=4, num_workers=0, method_args={"queue_size": 4},
            checkpoint=first.checkpoint_path,
        )
        assert (tmp_path / "run4b" / "checkpoints" / "last.ckpt").exists()

    def test_out_dir_conflict_raises(self, tmp_path, image_dir):
        out = tmp_path / "run5"
        pretrain(
            out=out, data=image_dir, model="torchvision/resnet18",
            epochs=1, batch_size=4, num_workers=0, method_args={"queue_size": 4},
        )
        with pytest.raises(ValueError, match="not empty"):
            pretrain(
                out=out, data=image_dir, model="torchvision/resnet18",
                epochs=1, batch_size=4, num_workers=0,
                method_args={"queue_size": 4},
            )

    def test_unknown_method_raises(self, tmp_path, image_dir):
        with pytest.raises(ValueError, match="Unknown method"):
            pretrain(
                out=tmp_path / "run6", data=image_dir,
                model="torchvision/resnet18", method="nope",
                epochs=1, batch_size=4, num_workers=0,
            )

    def test_unknown_key_in_method_args(self, tmp_path, image_dir):
        from autovision.errors import ConfigValidationError

        with pytest.raises(ConfigValidationError, match="Unknown key.*Did you mean"):
            pretrain(
                out=tmp_path / "run7", data=image_dir,
                model="torchvision/resnet18", epochs=1, batch_size=4,
                num_workers=0, method_args={"queue_szie": 16},
            )


class TestIntrospection:
    def test_list_models(self):
        models = list_models()
        assert "torchvision/resnet18" in models
        assert models == sorted(models)

    def test_list_methods(self):
        assert "distillation" in list_methods()
