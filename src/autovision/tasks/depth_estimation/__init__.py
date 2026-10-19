# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Depth estimation task (monocular depth regression).

Planned family layout (design doc 30, task-first tree). One directory per
model family, each following the standard four-module template:

    <family>/
        __init__.py      # public re-exports: <Family>DepthEstimation,
                         # <Family>DepthEstimationTrain(+Args)
        config.py        # dataclass train args + family defaults
        task_model.py    # inference/export-facing TaskNetwork
        train_model.py   # TrainModel subclass (training step, optimizer)

Adding a family: create the four modules, register the model key in the
registry/CLI dispatch, add a metrics module under
``autovision/metrics/depth_estimation.py`` (e.g. AbsRel, RMSE, δ<1.25),
and a CLI subcommand under ``cli/train_task/``. No families yet.
"""
