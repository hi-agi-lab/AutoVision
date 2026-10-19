# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Fault injection: SIGKILL mid-training, then resume from the checkpoint.

Complements the in-process atomicity tests: the child process is killed
without any cleanup, so only crash-safe design (atomic checkpoint rename,
resume path, mmap cache recovery) can produce a green outcome. Runs the
real ``pretrain`` entry point in a fresh interpreter, like an operator
would after a crashed job.
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from autovision.runtime.checkpoint import Checkpoint

pytestmark = [
    pytest.mark.filterwarnings("ignore::UserWarning"),
    pytest.mark.skipif(os.name != "posix", reason="SIGKILL fault injection"),
]

_CHILD_SCRIPT = """
import sys

from autovision import pretrain

resume = len(sys.argv) > 4 and sys.argv[4] == "resume"
pretrain(
    out=sys.argv[1],
    data=sys.argv[2],
    model="torchvision/resnet18",
    epochs=int(sys.argv[3]),
    batch_size=4,
    num_workers=0,
    method_args={"queue_size": 4},
    resume_interrupted=resume,
)
print("TRAINING_FINISHED", flush=True)
"""

_POLL_INTERVAL_S = 0.05
_PROCESS_TIMEOUT_S = 300.0


def _child_argv(out: Path, data: Path, epochs: int, resume: bool) -> list[str]:
    argv = [sys.executable, "-c", _CHILD_SCRIPT, str(out), str(data), str(epochs)]
    if resume:
        argv.append("resume")
    return argv


def _spawn(out: Path, data: Path, epochs: int, resume: bool, log_dir: Path):
    # Child output goes to files, not pipes: an unread pipe would block the
    # child once its buffer fills, stalling the kill-window polling loop.
    stdout = (log_dir / "child.out").open("w")
    stderr = (log_dir / "child.err").open("w")
    return subprocess.Popen(
        _child_argv(out, data, epochs, resume),
        stdout=stdout,
        stderr=stderr,
        text=True,
    ), stdout, stderr


def test_sigkill_mid_training_then_resume(tmp_path):
    """kill -9 between epochs must leave a loadable checkpoint the next
    process can resume from, with no manual cleanup."""
    from tests.conftest import create_images

    data = tmp_path / "images"
    create_images(data, num_images=8, size=(48, 48))
    out = tmp_path / "run"
    checkpoint = out / "checkpoints" / "last.ckpt"

    # Phase 1: train to the first epoch checkpoint, then SIGKILL.
    child, child_out, child_err = _spawn(
        out, data, epochs=2, resume=False, log_dir=tmp_path
    )
    try:
        deadline = time.monotonic() + _PROCESS_TIMEOUT_S
        while time.monotonic() < deadline:
            if checkpoint.exists():
                break
            if child.poll() is not None:
                child_out.close()
                child_err.close()
                pytest.fail(
                    "child exited (rc=%s) before checkpointing: %s"
                    % (child.returncode, (tmp_path / "child.err").read_text()[-2000:])
                )
            time.sleep(_POLL_INTERVAL_S)
        else:
            pytest.fail("checkpoint never appeared")
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.wait(timeout=60)
        child_out.close()
        child_err.close()

    # The killed run's checkpoint must be fully readable (atomic rename).
    killed_metadata = Checkpoint.from_path(checkpoint).autovision
    assert killed_metadata.epoch in (0, 1)  # epoch boundary or mid-epoch-2

    # Phase 2: a fresh process resumes without any manual repair.
    resumed = subprocess.run(
        _child_argv(out, data, epochs=2, resume=True),
        capture_output=True,
        text=True,
        timeout=_PROCESS_TIMEOUT_S,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "TRAINING_FINISHED" in resumed.stdout

    final_metadata = Checkpoint.from_path(checkpoint).autovision
    assert final_metadata.epoch == 1  # the 2-epoch budget completed
    assert final_metadata.global_step == 4  # 2 steps/epoch x 2 epochs
    assert final_metadata.global_step >= killed_metadata.global_step
