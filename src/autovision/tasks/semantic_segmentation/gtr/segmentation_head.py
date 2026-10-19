# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""FCN semantic-segmentation head on the GTR encoder's stride-8 level.

Re-port from Intellindust-AI-Lab GTR (``engine/gtr/semseg/gtrsemseg.py``,
class ``SemSegHead``, MIT License, Copyright (c) 2026 The GTR Authors;
the head itself is the mmseg ``FCNHead`` convention — conv3x3-BN-ReLU →
dropout → conv1x1 — which their base config SyncBN-converts), adapted by
AutoVision per MIT sections 2-4 (registration cleanup, typing); provenance ledger entry (GTR multi-task integration, roadmap design
doc 31 §11 S1).

Module names (``conv``/``bn``/``act``/``dropout``/``classifier``) are kept
verbatim so the official ``head.*`` checkpoint keys load unchanged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["SemSegHead"]


class SemSegHead(nn.Module):
    """conv3x3-BN-ReLU -> dropout -> conv1x1 (per-pixel class logits)."""

    def __init__(
        self,
        in_channels: int = 192,
        hidden_dim: int = 192,
        num_classes: int = 19,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Named (not Sequential-indexed) so the BN params are ``head.bn.*``:
        # the optimizer's no-decay rules match on the module type/name, and
        # an anonymous ``conv.1.weight`` would silently take weight_decay.
        self.conv = nn.Conv2d(
            in_channels, hidden_dim, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.act(self.bn(self.conv(feat)))))
