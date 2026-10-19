# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Multi-process DDP end-to-end tests launched through torchrun.

Closes the gap documented in booklet 12 §3 item 1: single-process DDP
building blocks are covered elsewhere (sampler epoch advance, metric state
gather, loader rebuild, forward alias); these tests run the *whole* stack
under a real ``torchrun`` launcher (WORLD_SIZE=2, gloo CPU backend) and
verify process-group init, distributed sampling, rank-zero-only artifacts
and run-to-run determinism.

Gated behind ``AUTOVISION_DDP_TESTS=1`` so the default suite stays fast;
the CI job for multi-process DDP sets the flag.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("AUTOVISION_DDP_TESTS") != "1",
    reason="set AUTOVISION_DDP_TESTS=1 to run torchrun multi-process DDP tests",
)

NPROC = 2


def _make_image_tree(root: Path, classes: int = 2, per_class: int = 4) -> Path:
    """Deterministic ImageFolder tree (PIL only, no torchvision needed)."""
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(0)
    for cls in range(classes):
        class_dir = root / f"class_{cls}"
        class_dir.mkdir(parents=True)
        for index in range(per_class):
            array = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            Image.fromarray(array).save(class_dir / f"img_{index:02d}.png")
    return root


def _write_pretrain_worker(path: Path) -> None:
    path.write_text(
        """
import json
import os
import sys
from pathlib import Path

from autovision import pretrain

out = Path(sys.argv[1])
data = Path(sys.argv[2])
result = pretrain(
    out=out,
    data=data,
    model="torchvision/resnet18",
    epochs=2,
    batch_size=4,
    num_workers=0,
    seed=0,
    method_args={"queue_size": 4},
)
rank = int(os.environ.get("RANK", "0"))
(out / f"rank_{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "epochs_run": result.epochs_run,
            "global_step": result.global_step,
        }
    )
)
""",
        encoding="utf-8",
    )


def _write_task_worker(path: Path) -> None:
    path.write_text(
        """
import json
import os
import sys
from pathlib import Path

from autovision import train_image_classification

out = Path(sys.argv[1])
train_root = Path(sys.argv[2])
val_root = Path(sys.argv[3])
result = train_image_classification(
    out=out,
    data={"train": train_root, "val": val_root},
    model="torchvision/resnet18",
    steps=4,
    batch_size=4,
    num_workers=0,
    seed=0,
    model_args={"image_size": (32, 32), "lr_warmup_steps": 2},
    trainer_args={"val_every_num_steps": 2},
)
rank = int(os.environ.get("RANK", "0"))
(out / f"rank_{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "best_metrics": result.best_metrics,
        }
    )
)
""",
        encoding="utf-8",
    )


def _run_torchrun(worker: Path, args: list[str], cwd: Path) -> str:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={NPROC}",
        str(worker),
        *args,
    ]
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES="",  # CPU + gloo: no accelerator requirement
        OMP_NUM_THREADS="1",
        AUTOVISION_TELEMETRY="0",  # the real switch (default off regardless)
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
        env.pop(name, None)  # torchrun injects these itself
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        f"torchrun failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    return result.stdout + result.stderr


def _rank_reports(out: Path) -> list[dict]:
    return [
        json.loads((out / f"rank_{rank}.json").read_text())
        for rank in range(NPROC)
    ]


def _load_state_dict(ckpt: Path) -> dict:
    return torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]


class TestPretrainTorchrunDistributed:
    def test_pretrain_ddp_e2e_deterministic_rank0_artifacts(self, tmp_path):
        data = _make_image_tree(tmp_path / "data")
        worker = tmp_path / "_pretrain_worker.py"
        _write_pretrain_worker(worker)

        _run_torchrun(worker, [str(tmp_path / "run1"), str(data)], tmp_path)
        _run_torchrun(worker, [str(tmp_path / "run2"), str(data)], tmp_path)

        for run in ("run1", "run2"):
            out = tmp_path / run
            reports = _rank_reports(out)
            # Both ranks trained the sharded dataset end to end.
            assert {r["rank"] for r in reports} == {0, 1}
            assert all(r["world_size"] == NPROC for r in reports)
            assert all(r["epochs_run"] == 2 for r in reports)
            # 8 images / 2 ranks / per-device batch 2 = 2 steps per epoch.
            assert all(r["global_step"] == 4 for r in reports)
            # Rank-zero-only artifacts.
            checkpoint = out / "checkpoints" / "last.ckpt"
            exported = out / "exported_models" / "exported_last.pt"
            assert checkpoint.is_file()
            assert exported.is_file()

        # Deterministic: two seeded multi-process runs produce identical
        # model state (DDP keeps replicas synchronized; queue methods are
        # per-rank by construction but seeded identically).
        first = _load_state_dict(tmp_path / "run1" / "checkpoints" / "last.ckpt")
        second = _load_state_dict(tmp_path / "run2" / "checkpoints" / "last.ckpt")
        assert set(first) == set(second)
        for key, tensor in first.items():
            assert torch.equal(tensor, second[key]), key


class TestTaskTorchrunDistributed:
    def test_classification_ddp_e2e_consistent_best_decision(self, tmp_path):
        train_root = _make_image_tree(tmp_path / "train")
        val_root = _make_image_tree(tmp_path / "val")
        worker = tmp_path / "_task_worker.py"
        _write_task_worker(worker)

        _run_torchrun(
            worker,
            [str(tmp_path / "run1"), str(train_root), str(val_root)],
            tmp_path,
        )
        _run_torchrun(
            worker,
            [str(tmp_path / "run2"), str(train_root), str(val_root)],
            tmp_path,
        )

        for run in ("run1", "run2"):
            out = tmp_path / run
            reports = _rank_reports(out)
            assert {r["rank"] for r in reports} == {0, 1}
            assert all(r["world_size"] == NPROC for r in reports)
            # Every rank computes the same aggregated validation metrics
            # (broadcast_object_list keeps best decisions consistent).
            top1 = [r["best_metrics"]["val_metric/accuracy_top1"] for r in reports]
            assert top1[0] == top1[1]
            assert 0.0 <= top1[0] <= 1.0
            assert (out / "checkpoints" / "last.ckpt").is_file()
            assert (out / "exported_models" / "exported_last.pt").is_file()

        # Deterministic across seeded multi-process runs.
        first = _rank_reports(tmp_path / "run1")
        second = _rank_reports(tmp_path / "run2")
        assert [r["best_metrics"] for r in first] == [
            r["best_metrics"] for r in second
        ]


def test_nccl_smoke_script_gloo(tmp_path):
    """The operator smoke script works under torchrun (gloo on CPU)."""
    output = _run_torchrun(
        Path(__file__).resolve().parents[1] / "scripts" / "nccl_smoke.py",
        ["--backend", "gloo"],
        tmp_path,
    )
    assert "all_reduce=" in output
    assert "OK" in output


def _write_fsdp_detection_worker(path: Path) -> None:
    path.write_text(
        """
import json
import sys
from pathlib import Path

from tests.test_train_object_detection import (
    data_config,
    model_args,
    transform_args,
)

from autovision import train_object_detection

out = Path(sys.argv[1])
train_json = Path(sys.argv[2])
val_json = Path(sys.argv[3])
images = Path(sys.argv[4])
resume = sys.argv[5] == "resume"
steps = int(sys.argv[6])
result = train_object_detection(
    out=out,
    data=data_config(train_json, val_json, images),
    steps=steps,
    batch_size=2,
    gradient_accumulation_steps=2,
    num_workers=0,
    devices="auto",
    precision="bf16-mixed",
    model_args={**model_args(), "lr_warmup_steps": 1},
    transform_args=transform_args(),
    save_checkpoint_args={
        "save_every_num_steps": 1,
        "save_last": True,
        "save_best": True,
    },
    trainer_args={"val_every_num_steps": 1, "parallel": "fsdp"},
    resume_interrupted=resume,
    overwrite=False,
)
rank = int(__import__("os").environ.get("RANK", "0"))
(out / f"rank_{rank}.json").write_text(
    json.dumps(
        {
            "rank": rank,
            "world_size": int(__import__("os").environ.get("WORLD_SIZE", "1")),
            "steps_run": result.steps_run,
            "best_metrics": result.best_metrics,
        }
    )
)
""",
        encoding="utf-8",
    )


def _run_torchrun_cuda(worker: Path, args: list[str], cwd: Path) -> str:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={NPROC}",
        str(worker),
        *args,
    ]
    env = dict(os.environ)
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env.update(
        OMP_NUM_THREADS="1",
        AUTOVISION_TELEMETRY="0",
        # The CI/host may expose one GPU; allow the 2-rank FSDP smoke to
        # shard across ranks sharing it. Real multi-GPU hosts do not need it.
        NCCL_MULTI_RANK_GPU_ENABLE="1",
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "GROUP_RANK", "ROLE_RANK"):
        env.pop(name, None)
    result = subprocess.run(
        command, cwd=cwd, env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, (
        f"torchrun FSDP failed (rc={result.returncode}):\n"
        f"--- stdout ---\n{result.stdout[-4000:]}\n"
        f"--- stderr ---\n{result.stderr[-4000:]}"
    )
    return result.stdout + result.stderr


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP needs CUDA")
class TestFsdpTorchrunDistributed:
    def test_fsdp_detection_checkpoint_export_resume(self, tmp_path):
        from tests.test_train_object_detection import make_coco_dataset

        worker = tmp_path / "_fsdp_worker.py"
        _write_fsdp_detection_worker(worker)
        train_json, val_json, images = make_coco_dataset(tmp_path / "data")
        out = tmp_path / "fsdp"
        _run_torchrun_cuda(
            worker,
            [str(out), str(train_json), str(val_json), str(images), "fresh", "2"],
            tmp_path,
        )

        reports = _rank_reports(out)
        assert {r["rank"] for r in reports} == {0, 1}
        assert all(r["world_size"] == NPROC for r in reports)
        assert all(r["steps_run"] == 2 for r in reports)
        assert (out / "checkpoints" / "last.ckpt").is_file()
        exported = out / "exported_models" / "exported_last.pt"
        assert exported.is_file()

        # The export is a normal task-model artifact (full, unsharded state).
        from autovision import load_task_model

        net = load_task_model(exported, device="cpu")
        assert net is not None

        # Resume from the FSDP full-state checkpoint on every rank.
        _run_torchrun_cuda(
            worker,
            [str(out), str(train_json), str(val_json), str(images), "resume", "4"],
            tmp_path,
        )
        resumed = _rank_reports(out)
        assert all(r["steps_run"] == 2 for r in resumed)
