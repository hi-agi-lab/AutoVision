#!/usr/bin/env python3
# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Image classification inference: top-1 label -> annotated image.

There is no released task checkpoint in the catalog. Pass an exported ``.pt``
from ``classification_train.py`` and optionally a class-name JSON:

    python examples/tasks/classification.py \
        --weights runs/examples/classification_train/exported_models/exported_last.pt \
        --class-names class_names.json
"""

from __future__ import annotations

from _common import (
    build_inference_parser,
    load_class_names,
    predict_and_annotate,
    resolve_device,
)

import autovision


def main() -> None:
    parser = build_inference_parser(
        "Image classification inference (exported .pt)",
        default_out="runs/examples/classification",
    )
    args = parser.parse_args()

    if args.weights is None:
        raise SystemExit(
            "Classification has no released task checkpoint. Train once with "
            "`python examples/tasks/classification_train.py` and pass "
            "--weights .../exported_models/exported_last.pt."
        )
    model = autovision.load_model(args.weights, device=resolve_device(args.device))
    names = load_class_names(args.class_names)
    predict_and_annotate(model, args, names, title="classification")


if __name__ == "__main__":
    main()
