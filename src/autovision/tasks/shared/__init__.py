# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Cross-task shared code, two kinds:

- flat modules: task-agnostic training infrastructure (schedules, EMA,
  deployment tiling);
- subpackages: architecture cores serving several task directories
  (``eomt/`` — the Encoder-only Mask Transformer behind the three
  segmentation tasks).

Nothing here may import from a single task package.
"""
