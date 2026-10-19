# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Open-vocabulary text assets and heads (design doc 27).

The language side of the AV-DETR open-vocab extension, kept in one module:

- **Vocabulary assets**: per-class prompt encoding through a FROZEN
  text tower (SigLIP / CLIP), computed once per (tower, vocabulary,
  templates) tuple and cached under ``Env.CACHE_DIR`` — training and
  inference never run the tower again. The ``identity`` encoder is a
  deterministic hash embedder for hermetic tests and offline smoke runs
  (never a research option).
- **:class:`ContrastiveClassHead`**: the reduced-view text head — query
  features are projected into the text embedding space, and the emitted
  ``(..., num_classes)`` logits are ``logit_scale * cosine + class_bias``
  (one vote per class). Every downstream consumer (criterion, matcher,
  decode, Dome dynamic selection) reads them exactly like baseline class
  logits, so the "reduce-then-reuse" red line of design doc 27 §3.2 holds.
- **:class:`TextCrossAttention`**: the zero-init gated text cross-attention
  sublayer injected into the decoder layers (v2). With the gate at its
  zero init the sublayer is a bitwise identity, so enabling the branch
  never disturbs a warm start (doc 27 §3.6).

The text towers are transient: they are built only inside
:func:`encode_class_names` (under ``torch.random.fork_rng``), never
registered on a task model — they stay out of ``state_dict`` / EMA /
optimizer / export by construction. The decoder's persistent
``ov_text_embeds`` buffer is the only language artifact that ships.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from autovision.runtime.env import Env

__all__ = [
    "VISDRONE_CLASS_NAMES",
    "ContrastiveClassHead",
    "TextCrossAttention",
    "class_bias_init",
    "encode_class_names",
    "resolve_text_embed_dim",
    "text_cache_path",
]

logger = logging.getLogger("autovision")


#: VisDrone2019-DET instance classes in the canonical dataset order. The
#: repo carries no dataset category list (the runtime ``categories`` come
#: from the dataset), so the open-vocab branch keeps the canonical mapping
#: here for prompt assets; ``class_names`` on the args stays the source of
#: truth and is cross-checked against the dataset at build time.
VISDRONE_CLASS_NAMES: tuple[str, ...] = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)


_TEXT_ENCODER_SPECS: dict[str, dict[str, object]] = {
    # repo_id: the HF checkpoint the frozen tower loads through
    # load_pretrained_verified (sha256 pinning is gate G1 of doc 27 §6;
    # unpinned repos fall back to the hub with a warning). embed_dim is
    # the tower's projected text width (the ContrastiveClassHead target).
    "siglip-base": {
        "repo_id": "google/siglip-base-patch16-224",
        "embed_dim": 768,
        "family": "siglip",
    },
    "clip-b32": {
        "repo_id": "openai/clip-vit-base-patch32",
        "embed_dim": 512,
        "family": "clip",
    },
    "clip-b16": {
        "repo_id": "openai/clip-vit-base-patch16",
        "embed_dim": 512,
        "family": "clip",
    },
    "identity": {
        "repo_id": None,
        "embed_dim": 64,
        "family": "identity",
    },
}


def resolve_text_embed_dim(text_encoder: str) -> int:
    """Projected text width of the given encoder."""
    try:
        return int(_TEXT_ENCODER_SPECS[text_encoder]["embed_dim"])
    except KeyError:
        supported = ", ".join(sorted(_TEXT_ENCODER_SPECS))
        raise ValueError(
            f"Unknown open-vocab text encoder '{text_encoder}'. "
            f"Supported: {supported}."
        ) from None


def class_bias_init(prior_prob: float = 0.01) -> float:
    """The baseline score-head bias init (``bias_init_with_prob(0.01)``).

    The contrastive head has no Linear bias; the per-class bias restores
    the baseline's early calibration so matched-pair focal costs start in
    the same regime (doc 27 §3.4).
    """
    return float(-math.log((1 - prior_prob) / prior_prob))


class ContrastiveClassHead(nn.Module):
    """Text-contrastive classification head (doc 27 §3.4).

    Emits the ``(B, Q, num_classes)`` reduced view consumed unchanged by
    the criterion/matcher/decode stack. ``text_embeds`` is passed in per
    call (the decoder owns the single persistent buffer — per-layer heads
    share it without duplicating tensors).

    The head is deliberately bias-free in the Linear sense: calibration
    lives in the per-class ``class_bias`` (init = the baseline score-head
    bias), so a fresh class can enter the vocabulary with the baseline's
    prior while the similarity path stays vocabulary-agnostic.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        text_dim: int,
        logit_scale_init: float = 2.659,
        logit_scale_max: float = 4.605,
        use_class_bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.logit_scale_max = float(logit_scale_max)
        self.proj = nn.Linear(in_dim, text_dim)
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))
        self.class_bias: nn.Parameter | None
        if use_class_bias:
            self.class_bias = nn.Parameter(
                torch.full((num_classes,), class_bias_init())
            )
        else:
            self.class_bias = None

    def reset_calibration(self, bias: float) -> None:
        """Set the per-class bias (the decoder ``_reset_parameters`` hook
        replacing ``init.constant_(head.bias, ...)`` for this head type)."""
        if self.class_bias is not None:
            with torch.no_grad():
                self.class_bias.fill_(float(bias))

    def forward(self, features: Tensor, text_embeds: Tensor) -> Tensor:
        """``features`` (..., in_dim), ``text_embeds`` (num_classes, text_dim)
        L2-normalized → logits (..., num_classes)."""
        if text_embeds.dim() != 2 or text_embeds.shape[0] != self.num_classes:
            raise ValueError(
                f"open-vocab text_embeds must be ({self.num_classes}, d), got "
                f"{tuple(text_embeds.shape)}."
            )
        projected = F.normalize(self.proj(features), dim=-1)
        scale = self.logit_scale.clamp(max=self.logit_scale_max)
        logits = scale * projected @ text_embeds.T
        if self.class_bias is not None:
            logits = logits + self.class_bias
        return logits


class TextCrossAttention(nn.Module):
    """Zero-init gated text cross-attention sublayer (doc 27 §3.6).

    ``target = target + gate * MHA(q=target, k=v=text_embeds)`` with the
    gate initialized at 0: the sublayer is a bitwise identity at init
    (Flamingo/GDINO warm-start semantics), so decoder layers that gain it
    reproduce the baseline forward exactly until training moves the gate.
    ``kdim``/``vdim`` let the layer width differ from the text width (the
    official decoder's wide-tail layers interpolate the query width).
    """

    def __init__(self, d_model: int, text_dim: int, num_heads: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            kdim=text_dim,
            vdim=text_dim,
            batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, target: Tensor, text_embeds: Tensor) -> Tensor:
        query = self.q_proj(target)
        # The shared buffer is (num_classes, text_dim); MHA needs the
        # batched (B, num_classes, text_dim) key/value view.
        keys = text_embeds.unsqueeze(0).expand(target.shape[0], -1, -1)
        attended, _ = self.attn(query, keys, keys)
        return target + self.gate * attended


# ── vocabulary encoding ─────────────────────────────────────────────────


def _identity_embed(
    class_names: tuple[str, ...], template: str, dim: int
) -> Tensor:
    """Deterministic hash embeddings (hermetic tests / offline smoke).

    sha256 of "template//class name" → uniform vector in [-1, 1], L2
    normalized. No RNG, no network, stable across processes.
    """
    rows = torch.empty((len(class_names), dim))
    for index, name in enumerate(class_names):
        digest = hashlib.sha256(f"{template}//{name}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little")
        generator = torch.Generator().manual_seed(seed)
        rows[index] = torch.empty(dim).uniform_(-1.0, 1.0, generator=generator)
    return F.normalize(rows, dim=-1)


def _tower_weight_digest(repo_id: str) -> str:
    """Short digest of the pinned tower weights (doc 27 risk R6): a cache
    key must not outlive the weights it was encoded with — a re-pinned
    tower re-encodes instead of silently serving stale embeddings."""
    try:
        from autovision.encoders.weight_verification import get_weight_artifact

        artifact = get_weight_artifact(repo_id)
    except Exception:  # pragma: no cover - manifest lookup is total today
        return "unpinned"
    if artifact is None or not artifact.sha256:
        return "unpinned"
    # artifact.sha256 maps filename -> pinned hash; digest the sorted pairs
    # so the value is stable across processes and sensitive to any re-pin.
    payload = json.dumps(dict(sorted(artifact.sha256.items())), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def text_cache_path(
    text_encoder: str,
    class_names: tuple[str, ...],
    prompt_template: str,
    prompt_templates: tuple[str, ...] | None,
    cache_dir: Path | None = None,
) -> Path:
    """Cache file for one (tower weights, vocabulary, templates) tuple.

    The payload is hashed STRUCTURALLY (a JSON list of names) — separator
    characters inside class names cannot collide two vocabularies.
    """
    if cache_dir is None:
        cache_dir = Env.CACHE_DIR.value / "open_vocab"
    repo_id = _TEXT_ENCODER_SPECS[text_encoder]["repo_id"]
    weight_digest = (
        _tower_weight_digest(str(repo_id)) if repo_id is not None else "identity"
    )
    payload = json.dumps(
        {
            "encoder": text_encoder,
            "weights": weight_digest,
            "template": prompt_template,
            "templates": list(prompt_templates) if prompt_templates else None,
            "classes": list(class_names),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"ov_text_{text_encoder}_{digest}.pt"


def encode_class_names(
    class_names: tuple[str, ...],
    text_encoder: str = "siglip-base",
    prompt_template: str = "a photo of a {}.",
    prompt_templates: tuple[str, ...] | None = None,
    cache_embeddings: bool = True,
    cache_dir: Path | None = None,
) -> Tensor:
    """Encode one class-name vocabulary into L2-normalized fp32 embeddings.

    The real towers run ONCE per (encoder, vocabulary, templates) tuple:
    the result is cached under ``Env.CACHE_DIR`` and reloaded afterwards,
    so training and inference never pay the tower again. The tower build
    and forward run under ``torch.random.fork_rng`` — the transient RNG
    draw of ``from_pretrained``'s (re)initialization must not shift the
    task-model init stream (doc 27 §3.9, ssl-head precedent). The tower is
    never returned: it is dropped here, so it cannot leak into a module
    tree.
    """
    if not class_names:
        raise ValueError("open-vocab vocabulary is empty.")
    templates = tuple(prompt_templates) if prompt_templates else (prompt_template,)
    for template in templates:
        if "{}" not in template:
            raise ValueError(
                f"open-vocab prompt template {template!r} lacks the '{{}}' "
                f"class-name slot."
            )
    spec = _TEXT_ENCODER_SPECS.get(text_encoder)
    if spec is None:
        supported = ", ".join(sorted(_TEXT_ENCODER_SPECS))
        raise ValueError(
            f"Unknown open-vocab text encoder '{text_encoder}'. "
            f"Supported: {supported}."
        )

    cache_file = text_cache_path(
        text_encoder, class_names, prompt_template, prompt_templates, cache_dir
    )
    if cache_embeddings and cache_file.exists():
        try:
            embeds = torch.load(cache_file, map_location="cpu", weights_only=True)
        except Exception as error:
            logger.warning(
                "open-vocab text cache %s is unreadable (%s) — re-encoding.",
                cache_file,
                error,
            )
        else:
            if embeds.shape == (len(class_names), int(spec["embed_dim"])):
                return embeds
            logger.warning(
                "open-vocab text cache %s has shape %s (expected %s) — "
                "re-encoding.",
                cache_file,
                tuple(embeds.shape),
                (len(class_names), int(spec["embed_dim"])),
            )

    if spec["family"] == "identity":
        logger.warning(
            "open_vocab is using the 'identity' HASH embedder — embeddings "
            "carry no language alignment; research numbers from this config "
            "are meaningless (tests / offline smoke only, doc 27 D8)."
        )

    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        if spec["family"] == "identity":
            embeds = _identity_embed(class_names, prompt_template, int(spec["embed_dim"]))
        else:
            embeds = _encode_with_tower(
                class_names, templates, str(spec["repo_id"]), str(spec["family"])
            )
    elapsed = time.perf_counter() - started

    if cache_embeddings:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        # Atomic publish with a per-process temp name: on a cold cache
        # every DDP rank encodes concurrently, and a shared tmp path
        # could interleave truncate/write and poison the cache (doc 27
        # review P1-3). os.replace is atomic on POSIX.
        handle, tmp_name = tempfile.mkstemp(
            dir=cache_file.parent, prefix=f".{cache_file.stem}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "wb") as handle_file:
                torch.save(embeds, handle_file)
            os.replace(tmp_name, cache_file)
        except Exception:
            os.unlink(tmp_name)
            raise
    logger.info(
        "open-vocab: encoded %d class prompts with '%s' in %.1fs (%s)",
        len(class_names),
        text_encoder,
        elapsed,
        "cached" if cache_embeddings else "no cache",
    )
    return embeds


def _encode_with_tower(
    class_names: tuple[str, ...],
    templates: tuple[str, ...],
    repo_id: str,
    family: str,
) -> Tensor:
    """Run the frozen tower on CPU (one-time cost) and return the mean over
    templates of the L2-normalized projected text embeddings."""
    from autovision.encoders.weight_verification import load_pretrained_verified

    if family == "siglip":
        from transformers import SiglipModel

        model_cls: type = SiglipModel
        # SigLIP was trained with fixed-width padded batches (model-card
        # contract): padding to max_length is REQUIRED for aligned
        # embeddings. SigLIP has no projection matrix — its pooler_output
        # IS the shared-space embedding (sigmoid-loss design).
        tokenize_kwargs: dict[str, object] = {"padding": "max_length"}
    else:
        from transformers import CLIPModel

        model_cls = CLIPModel
        tokenize_kwargs = {"padding": True}

    model = load_pretrained_verified(model_cls, repo_id)
    model.eval()
    model.requires_grad_(False)
    tokenizer = _load_tokenizer(repo_id)

    per_template: list[Tensor] = []
    with torch.inference_mode():
        for template in templates:
            prompts = [template.format(name) for name in class_names]
            batch = tokenizer(
                prompts, truncation=True, return_tensors="pt", **tokenize_kwargs
            )
            # transformers v5: get_text_features returns the text tower's
            # ModelOutput — the aligned embedding sits in pooler_output
            # (CLIP: post text_projection; SigLIP: the raw shared space).
            output = model.get_text_features(**batch)
            features = getattr(output, "pooler_output", output)
            per_template.append(F.normalize(features.float(), dim=-1))
    return F.normalize(torch.stack(per_template).mean(dim=0), dim=-1).cpu()


def _load_tokenizer(repo_id: str):
    from transformers import AutoTokenizer

    # Tokenizer only (no weights); the model weights are license-gated
    # and sha256-pinned elsewhere.
    return AutoTokenizer.from_pretrained(repo_id)  # nosec B615
