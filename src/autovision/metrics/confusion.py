# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Detection confusion matrix built from error-analysis matches.

The last row is the false-positive row and the last column the
false-negative column. A misclassification FP (``fp_reason ==
"classification"``) is placed at ``[true_class, predicted_class]`` when the
matched GT class is known, so class-pair confusions are visible; every other
FP lands in the FP row under its predicted class.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from autovision.metrics.error_analysis import Match

__all__ = ["ConfusionMatrix"]


@dataclass
class ConfusionMatrix:
    """Integer matrix over ``(classes + FP) x (classes + FN)``."""

    matrix: np.ndarray
    class_ids: list[int]
    class_names: list[str]

    @property
    def class_count(self) -> int:
        return len(self.class_ids)

    @classmethod
    def from_matches(
        cls,
        matches: Sequence["Match"],
        *,
        class_names: Mapping[int, str] | None = None,
        class_ids: Sequence[int] | None = None,
    ) -> "ConfusionMatrix":
        """Build the matrix from ``match_detections`` rows.

        ``class_ids`` fixes the class order; otherwise every class seen in a
        TP/FN/FP row (including the GT class of a misclassification) is used.
        """
        names = dict(class_names or {})
        if class_ids is None:
            seen: set[int] = set(names)
            for match in matches:
                if match.kind == "ignored":
                    continue
                seen.add(int(match.class_id))
                if match.other_class_id is not None:
                    seen.add(int(match.other_class_id))
            ordered = sorted(seen)
        else:
            ordered = [int(value) for value in class_ids]
        index = {class_id: position for position, class_id in enumerate(ordered)}
        size = len(ordered) + 1
        matrix: np.ndarray = np.zeros((size, size), dtype=np.int64)
        fp_row = size - 1
        fn_col = size - 1

        for match in matches:
            if match.kind == "ignored":
                continue
            predicted = int(match.class_id)
            column = index.get(predicted)
            if match.kind == "tp":
                if column is not None:
                    matrix[column, column] += 1
            elif match.kind == "fn":
                if column is not None:
                    matrix[column, fn_col] += 1
            elif match.kind == "fp":
                other = match.other_class_id
                row = index.get(int(other)) if other is not None else None
                if row is not None and column is not None:
                    matrix[row, column] += 1
                elif column is not None:
                    matrix[fp_row, column] += 1
        return cls(
            matrix=matrix,
            class_ids=ordered,
            class_names=[names.get(class_id, str(class_id)) for class_id in ordered],
        )

    def to_dict(self) -> dict[str, Any]:
        """JSON-native form (names, ids, matrix, sentinel indexes)."""
        return {
            "classes": list(self.class_names),
            "class_ids": list(self.class_ids),
            "matrix": self.matrix.tolist(),
            "fp_row": self.class_count,
            "fn_col": self.class_count,
        }

    def to_markdown(self) -> str:
        """Markdown table with a ``FN`` column and an ``FP`` row."""
        if self.class_count == 0:
            return "_（无类别）_"
        header = ["True \\ Predicted", *self.class_names, "FN"]
        lines = [
            "| " + " | ".join(header) + " |",
            "|" + "---|" * len(header),
        ]
        for row_index, name in enumerate(self.class_names):
            values = [str(int(value)) for value in self.matrix[row_index]]
            lines.append("| " + " | ".join([name, *values]) + " |")
        fp_values = [str(int(value)) for value in self.matrix[-1]]
        lines.append("| " + " | ".join(["FP", *fp_values]) + " |")
        return "\n".join(lines)
