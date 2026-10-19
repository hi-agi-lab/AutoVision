#!/usr/bin/env bash
# Container smoke test: 8 synthetic images -> 1 epoch pretrain -> artifacts.
set -euo pipefail
python - << 'PY'
import sys, tempfile, pathlib
import numpy as np
from PIL import Image

tmp = pathlib.Path(tempfile.mkdtemp())
(tmp / "images").mkdir(parents=True)
rng = np.random.default_rng(0)
for i in range(8):
    array = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(array).save(tmp / "images" / f"img_{i:03d}.jpg")

import autovision
result = autovision.pretrain(
    out=tmp / "run", data=tmp / "images", model="torchvision/resnet18",
    epochs=1, batch_size=4, num_workers=0,
    method_args={"queue_size": 4, "teacher_pretrained": False},
)
assert result.checkpoint_path.exists() and result.exported_model_path.exists()
print("smoke test OK:", result.out_dir)
PY
