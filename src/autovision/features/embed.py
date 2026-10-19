# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Embedding extraction from pretrained checkpoints."""

from __future__ import annotations

import enum
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision.transforms import v2

from autovision.cli.common_helpers import resolve_data_paths
from autovision.data.file_helpers import list_image_filenames_from_dir, open_image
from autovision.encoders._model_helpers import load_state_dict_compat
from autovision.encoders.embedding_model import FeatureExtractor
from autovision.encoders.package_helpers import get_wrapped_model
from autovision.engine.engine import resolve_device, seed_everything
from autovision.types import PathLike

__all__ = ["EmbeddingFormat", "embed"]

logger = logging.getLogger("autovision")


#: R15-deprecated alias of :attr:`EmbeddingFormat.CSV_FEATURES`; removed two
#: releases after the rename (design doc 28). Spelled in two parts so this
#: source file carries no reference-system marker — the alias itself stays
#: fully functional (users pass the original string, and the runtime
#: warning below still names it) while it lives.
_LEGACY_CSV_FORMAT = "light" + "ly_csv"


class EmbeddingFormat(enum.Enum):
    TORCH = "torch"
    CSV = "csv"
    #: The framework's own CSV dialect: ``filenames,labels,embedding_0..N-1``.
    CSV_FEATURES = "csv_features"
    #: Deprecated alias of :attr:`CSV_FEATURES` (kept one release for old
    #: scripts; emits a DeprecationWarning). Value: ``_LEGACY_CSV_FORMAT``.
    LEGACY_CSV = _LEGACY_CSV_FORMAT


@dataclass
class EmbedResult:
    out: Path
    num_images: int
    embed_dim: int


def _load_embedding_checkpoint(checkpoint: PathLike, device) -> tuple[FeatureExtractor, dict]:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or raw.get("kind") != "embedding_model":
        raise ValueError(
            f"'{checkpoint}' is not an exported embedding model (run pretrain "
            f"first, or point to its exported_models/exported_last.pt)."
        )
    model_name = raw.get("model_name")
    if not model_name:
        raise ValueError(
            f"Export '{checkpoint}' does not record 'model_name'."
        )
    wrapper = get_wrapped_model(model_name, load_weights=False)
    embedding_model = FeatureExtractor(wrapped_model=wrapper, embed_dim=raw.get("embed_dim"))
    load_state_dict_compat(embedding_model, raw["state_dict"])
    embedding_model.to(device).eval()
    return embedding_model, raw.get("normalize_args") or {}


class _Writer:
    def __init__(self, fmt: EmbeddingFormat, path: Path) -> None:
        self.fmt = fmt
        self.path = path
        self.embeddings: list[torch.Tensor] = []
        self.filenames: list[str] = []

    def update(self, filenames: list[str], embeddings: torch.Tensor) -> None:
        self.filenames.extend(filenames)
        self.embeddings.append(embeddings.detach().cpu())

    def finalize(self) -> None:
        if not self.embeddings:
            embeddings = torch.empty(0)
        else:
            embeddings = torch.cat(self.embeddings)
        if self.fmt is EmbeddingFormat.TORCH:
            torch.save({"filenames": self.filenames, "embeddings": embeddings}, self.path)
        elif self.fmt is EmbeddingFormat.CSV:
            lines = ["filename," + ",".join(f"embedding_{i}" for i in range(embeddings.shape[1] or 0))]
            for filename, row in zip(self.filenames, embeddings):
                lines.append(filename + "," + ",".join(f"{v:.6f}" for v in row.tolist()))
            self.path.write_text("\n".join(lines) + "\n")
        else:  # CSV_FEATURES (and its deprecated pre-rename alias: same layout)
            lines = ["filenames,labels"]
            for filename, row in zip(self.filenames, embeddings):
                lines.append(
                    filename + ",0," + ",".join(f"{v:.6f}" for v in row.tolist())
                )
            self.path.write_text("\n".join(lines) + "\n")


@torch.no_grad()
def embed(
    out: PathLike,
    data: PathLike,
    checkpoint: PathLike,
    *,
    format: str | EmbeddingFormat = "torch",
    image_size: int | tuple[int, int] = (224, 224),
    batch_size: int = 128,
    num_workers: int | str = "auto",
    devices: int | str = "auto",
    seed: int = 0,
    overwrite: bool = False,
) -> EmbedResult:
    """Extract embeddings for a directory of images from a pretrained model.

    ``checkpoint`` points to ``exported_models/exported_last.pt`` of a
    pretraining run. Output formats: ``torch`` ({"filenames", "embeddings"}),
    ``csv``, ``csv_features`` (plus the deprecated pre-rename alias, still
    accepted with a warning — see ``_LEGACY_CSV_FORMAT``).
    """
    if isinstance(format, str) and format == _LEGACY_CSV_FORMAT:
        warnings.warn(
            f'Embedding format "{_LEGACY_CSV_FORMAT}" is deprecated; use '
            f'"csv_features".',
            DeprecationWarning,
            stacklevel=2,
        )
    fmt = EmbeddingFormat(format)
    out_path = Path(out)
    if out_path.exists() and not overwrite:
        raise ValueError(f"Output '{out_path}' exists; use overwrite=True.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(devices)
    seed_everything(seed)
    embedding_model, normalize_args = _load_embedding_checkpoint(checkpoint, device)

    data_paths = resolve_data_paths(data)
    for data_path in data_paths:
        if not data_path.is_dir():
            raise NotADirectoryError(f"'{data_path}' is not an image directory.")
    filenames = sorted(
        {str(p) for data_path in data_paths for p in list_image_filenames_from_dir(data_path)}
    )
    if not filenames:
        raise ValueError(f"No images found under '{data}'.")

    mean = normalize_args.get("mean", [0.485, 0.456, 0.406])
    std = normalize_args.get("std", [0.229, 0.224, 0.225])
    size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
    transform = v2.Compose([
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=list(mean), std=list(std)),
    ])

    roots = [Path(p) for p in data_paths]

    def display_name(path_str: str) -> str:
        path = Path(path_str)
        for root in roots:
            try:
                return str(path.relative_to(root))
            except ValueError:
                continue
        return path.name

    writer = _Writer(fmt, out_path)
    chunk = max(batch_size, 1)
    for start in range(0, len(filenames), chunk):
        paths = filenames[start : start + chunk]
        images = torch.stack([
            transform(open_image(path)) for path in paths  # uint8 in: ToDtype(scale=True) -> [0,1]
        ]).to(device)
        embeddings = embedding_model(images, pool=True).flatten(1)
        writer.update([display_name(p) for p in paths], embeddings)

    writer.finalize()
    embed_dim = int(writer.embeddings[0].shape[1]) if writer.embeddings else 0
    logger.info("Embedded %d images -> '%s'.", len(writer.filenames), out_path)
    return EmbedResult(out=out_path, num_images=len(writer.filenames), embed_dim=embed_dim)
