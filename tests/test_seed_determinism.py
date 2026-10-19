# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Seed determinism of the public training entry points.

Regression for the late-seeding defect found on 2026-10-16: the entries
built the model/method (random init) *before* the engine's ``fit`` applied
``seed_everything``, so two same-seed runs produced different weights. The
fix seeds at the top of each entry's assembly (``cli/train.py`` for
pretraining, ``cli/train_task/*`` for the task families).

These tests are deliberately un-gated (unlike the torchrun DDP suite): two
same-seed, single-process runs must produce bit-identical checkpoints.
"""

from __future__ import annotations

import hashlib

import torch

from autovision import pretrain, train_image_classification


def _image_tree(root, classes=2, per_class=3):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    for cls in range(classes):
        class_dir = root / f"class_{cls}"
        class_dir.mkdir(parents=True, exist_ok=True)
        for index in range(per_class):
            array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(array).save(class_dir / f"img_{index:02d}.png")
    return root


def _state_hash(checkpoint) -> str:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].cpu().numpy().tobytes())
    return digest.hexdigest()


def test_pretrain_two_seeded_runs_are_bit_identical(tmp_path):
    data = _image_tree(tmp_path / "data")
    hashes = []
    for run in ("run1", "run2"):
        pretrain(
            out=tmp_path / run,
            data=data,
            model="torchvision/resnet18",
            epochs=1,
            batch_size=2,
            num_workers=0,
            seed=0,
            method_args={"queue_size": 4},
        )
        hashes.append(_state_hash(tmp_path / run / "checkpoints" / "last.ckpt"))
    assert hashes[0] == hashes[1]


def test_task_training_two_seeded_runs_are_bit_identical(tmp_path):
    train_root = _image_tree(tmp_path / "train")
    val_root = _image_tree(tmp_path / "val")
    hashes = []
    for run in ("run1", "run2"):
        train_image_classification(
            out=tmp_path / run,
            data={"train": str(train_root), "val": str(val_root)},
            model="torchvision/resnet18",
            steps=2,
            batch_size=2,
            num_workers=0,
            seed=0,
        )
        hashes.append(_state_hash(tmp_path / run / "checkpoints" / "last.ckpt"))
    assert hashes[0] == hashes[1]
