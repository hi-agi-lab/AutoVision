# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""In-training kNN probe over frozen embeddings (design doc §7.5).

The pretraining loop's only quality signal is ``train_loss``; the kNN probe
adds an absolute one: periodically embed every image once, classify a query
split by majority vote of its ``k`` nearest gallery neighbors (cosine
similarity), and log the accuracy. Evaluation is fully detached from the
training path — it never touches the dataloader, optimizer, or gradients.

Labels come from the dataset layout: the *parent directory name* of each
image is its class (ImageFolder convention). A flat directory therefore has
no labels and the probe stays off unless ``enabled=True`` is forced.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Sequence

import torch
from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from autovision.augment.transform_args import MethodTransformArgs
from autovision.config.base import PydanticConfig

if TYPE_CHECKING:
    from autovision.pretext.recipe import PretextRecipe

__all__ = ["KnnEvalArgs", "KnnEvalRunner", "knn_accuracy"]

logger = logging.getLogger("autovision")

# ``enabled="auto"`` turns the probe on for datasets up to this size.
_AUTO_ENABLE_MAX_DATASET_SIZE = 500_000


class KnnEvalArgs(PydanticConfig):
    """Arguments for the in-training kNN probe (``method_args.knn_eval``)."""

    enabled: bool | Literal["auto"] = "auto"
    every_n_epochs: int = 1
    k: int = 10
    split: float = 0.5
    batch_size: int = 256
    max_samples: int = 100_000
    # True once "auto" was resolved to a bool by resolve_auto_fields(); an
    # auto-enabled probe degrades silently on an unlabeled layout, while an
    # explicit enabled=True is a hard configuration error there.
    was_auto: bool = Field(default=False, exclude=True)

    def resolve_auto_fields(self, dataset_size: int | None = None) -> None:
        if self.enabled == "auto":
            if dataset_size is None:
                raise ValueError(
                    "knn_eval.enabled='auto' requires the dataset size."
                )
            self.enabled = dataset_size <= _AUTO_ENABLE_MAX_DATASET_SIZE
            self.was_auto = True


def labels_for_filenames(filenames: Sequence[str | Path]) -> list[str] | None:
    """Class label per filename (parent directory name), or ``None``.

    Returns ``None`` when the layout carries no class information: fewer
    than two distinct parent directories (flat folder / single class) or any
    unresolvable path.
    """
    labels: list[str] = []
    for filename in filenames:
        name = filename["filenames"] if isinstance(filename, dict) else str(filename)
        parent = Path(name).parent.name
        if not parent:
            return None
        labels.append(parent)
    if len(set(labels)) < 2:
        return None
    return labels


class _KnnProbeDataset(Dataset):
    """Single-view dataset yielding ``(embedding_input, label_index)``."""

    def __init__(
        self,
        filenames: Sequence[str | Path],
        labels: Sequence[str],
        transform_args: MethodTransformArgs,
    ) -> None:
        from autovision.data.file_helpers import open_image

        self._open_image = open_image
        self.filenames = filenames
        classes = sorted(set(labels))
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        self.label_indices = [self.class_to_index[label] for label in labels]
        height, width = transform_args.image_size
        from torchvision.transforms import v2

        mean, std = transform_args.normalize.mean, transform_args.normalize.std
        self.transform = v2.Compose(
            [
                v2.Resize(size=(height, width), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=list(mean), std=list(std)),
            ]
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        name = self.filenames[index]
        filename = name["filenames"] if isinstance(name, dict) else str(name)
        image = self._open_image(Path(filename))
        view = self.transform({"image": image})
        return view["image"], self.label_indices[index]


@torch.no_grad()
def _extract_embeddings(
    method: "PretextRecipe",
    dataset: Dataset,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    device = next(method.embedding_model.parameters()).device
    was_training = method.embedding_model.training
    method.embedding_model.eval()
    embeddings, label_indices = [], []
    try:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        for images, targets in loader:
            pooled = method.embedding_model(
                images.to(device, non_blocking=True), pool=True
            ).flatten(start_dim=1)
            embeddings.append(torch.nn.functional.normalize(pooled, p=2, dim=-1).cpu())
            label_indices.append(targets)
    finally:
        if was_training:
            method.embedding_model.train()
    return torch.cat(embeddings), torch.cat(label_indices)


def knn_accuracy(
    embeddings: Tensor,
    label_indices: Tensor,
    k: int,
    split: float,
) -> dict[str, float]:
    """Cosine kNN classification accuracy on a gallery/query split.

    Embeddings must already be L2-normalized. The first ``split`` fraction of
    a seeded shuffle forms the gallery (reference) set; queries are classified
    by majority vote over their ``k`` nearest gallery neighbors. Ties resolve
    to the smaller class index (``torch.bincount`` argmax order).
    """
    num_samples = embeddings.size(0)
    num_gallery = max(1, min(int(num_samples * split), num_samples - 1))
    k = max(1, min(k, num_gallery))
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(num_samples, generator=generator)
    gallery = permutation[:num_gallery]
    query = permutation[num_gallery:]
    if query.numel() == 0:
        return {"accuracy_top1": float("nan")}

    similarity = embeddings[query] @ embeddings[gallery].T  # cosine: normalized
    topk_indices = similarity.topk(k, dim=1).indices  # (Q, k)
    votes = label_indices[gallery][topk_indices]  # (Q, k)
    num_classes = int(label_indices.max().item()) + 1
    predictions = torch.stack(
        [torch.bincount(row, minlength=num_classes).argmax() for row in votes]
    )
    correct = (predictions == label_indices[query]).float()
    result = {"accuracy_top1": float(correct.mean())}
    if k >= 5:
        # A query counts as top-5 correct when any of its 5 neighbors matches.
        top5_hit = (votes[:, :5] == label_indices[query, None]).any(dim=1).float()
        result["accuracy_top5"] = float(top5_hit.mean())
    return result


class KnnEvalRunner:
    """Runs the kNN probe at epoch boundaries; owned by the engine.

    Built once by the ``pretrain`` command and attached to the method as
    ``method.knn_eval_runner``. The probe dataset is materialized lazily on
    the first run so disabled runs pay nothing.
    """

    def __init__(
        self,
        args: KnnEvalArgs,
        method: "PretextRecipe",
        filenames: Sequence[str | Path],
        transform_args: MethodTransformArgs,
    ) -> None:
        self.args = args
        self.method = method
        self.filenames = list(filenames)
        self.transform_args = transform_args
        self._dataset: _KnnProbeDataset | None = None

    def should_run(self, epoch: int) -> bool:
        return (epoch + 1) % max(1, self.args.every_n_epochs) == 0

    def _probe_dataset(self) -> _KnnProbeDataset:
        if self._dataset is None:
            filenames = self.filenames
            if len(filenames) > self.args.max_samples:
                step = len(filenames) / self.args.max_samples
                filenames = [
                    filenames[int(i * step)] for i in range(self.args.max_samples)
                ]
            labels = labels_for_filenames(filenames)
            if labels is None:
                raise ValueError(
                    "knn_eval is enabled but the dataset layout has no class "
                    "labels (expecting ImageFolder-style class subdirectories)."
                )
            self._dataset = _KnnProbeDataset(filenames, labels, self.transform_args)
        return self._dataset

    def run(self) -> dict[str, float]:
        dataset = self._probe_dataset()
        embeddings, label_indices = _extract_embeddings(
            self.method, dataset, batch_size=self.args.batch_size
        )
        metrics = knn_accuracy(embeddings, label_indices, k=self.args.k, split=self.args.split)
        metrics = {f"knn_eval/{name}": value for name, value in metrics.items()}
        logger.info("kNN probe: %s", metrics)
        return metrics


def attach_knn_eval_runner(
    method: "PretextRecipe",
    method_args: Any,
    filenames: Sequence[str | Path],
    transform_args: MethodTransformArgs,
) -> KnnEvalRunner | None:
    """Build and attach the runner if ``method_args.knn_eval`` is enabled.

    ``enabled=True`` with an unlabeled layout is a configuration error;
    ``enabled="auto"``/``True`` already resolved to a bool before this runs.
    """
    knn_args: KnnEvalArgs | None = getattr(method_args, "knn_eval", None)
    if knn_args is None or knn_args.enabled is False:
        return None
    has_labels = labels_for_filenames(filenames) is not None
    if not has_labels:
        if knn_args.enabled is True and not knn_args.was_auto:
            raise ValueError(
                "method_args.knn_eval.enabled=True requires an ImageFolder-style "
                "dataset (one subdirectory per class), but no class labels were "
                "found in the image paths."
            )
        logger.warning(
            "knn_eval was auto-enabled but the dataset layout has no class "
            "labels (expecting ImageFolder-style class subdirectories); "
            "disabling the probe."
        )
        return None
    if knn_args.every_n_epochs < 1:
        raise ValueError("knn_eval.every_n_epochs must be >= 1.")
    runner = KnnEvalRunner(
        args=knn_args, method=method, filenames=filenames, transform_args=transform_args
    )
    method.knn_eval_runner = runner  # type: ignore[attr-defined]
    return runner
