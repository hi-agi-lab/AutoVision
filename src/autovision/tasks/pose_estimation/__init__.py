# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Pose estimation task (keypoint-based, e.g. 2D human/body-part pose).

Planned family layout (design doc 30, task-first tree). One directory per
model family, each following the standard four-module template:

    <family>/
        __init__.py      # public re-exports: <Family>PoseEstimation,
                         # <Family>PoseEstimationTrain(+Args)
        config.py        # dataclass train args + family defaults
        task_model.py    # inference/export-facing TaskNetwork
        train_model.py   # TrainModel subclass (training step, optimizer)

Adding a family: create the four modules, register the model key in the
registry/CLI dispatch, add a metrics module under
``autovision/metrics/pose_estimation.py`` (e.g. PCK, OKS/AP), and a CLI
subcommand under ``cli/train_task/``. No families yet.
"""
