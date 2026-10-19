# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Tests for the open-vocabulary text-contrastive branch (design doc 27 §8).

All cases run offline against the tiny ``dinov2-vittest-notpretrained``
variant and the deterministic ``identity`` text embedder (hermetic — no
downloads, no network), mirroring the conventions of
``tests/test_detr_ssl.py`` / ``tests/test_detr_dome.py``.
"""


import pytest
import torch

from autovision.optim.optimizer_helpers import get_weight_decay_parameters
from autovision.tasks.detection.detr import (
    DETRDensityArgs,
    DETRObjectDetection,
    DETRObjectDetectionTrain,
    DETRObjectDetectionTrainArgs,
    DETROpenVocabArgs,
)
from autovision.tasks.detection.shared import open_vocab as ov
from autovision.tasks.detection.shared.open_vocab import (
    ContrastiveClassHead,
    TextCrossAttention,
    class_bias_init,
    encode_class_names,
    resolve_text_embed_dim,
    text_cache_path,
)
from autovision.tasks.detection.shared.rtdetrv2_decoder import (
    RTDETRv2Decoder,
    bias_init_with_prob,
)

VARIANT_ARGS = dict(
    backbone="dinov2-vittest-notpretrained",
    image_size=(224, 224),
    num_queries=12,
    num_top_queries=12,
)
VOCAB = ("car", "person", "bus")


def make_ov_args(**overrides) -> DETROpenVocabArgs:
    defaults: dict = dict(text_encoder="identity", class_names=VOCAB)
    defaults.update(overrides)
    return DETROpenVocabArgs(**defaults)


def make_task_model(decoder: str = "dfine", **ov_overrides) -> DETRObjectDetection:
    torch.manual_seed(0)
    return DETRObjectDetection(
        num_classes=3,
        model_name="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        decoder=decoder,
        num_queries=12,
        dec_layers=2,
        num_denoising=4,
        open_vocab=make_ov_args(**ov_overrides),
    )


def make_train_model(decoder: str = "dfine", **overrides) -> DETRObjectDetectionTrain:
    torch.manual_seed(0)
    args_kwargs = dict(
        num_classes=3,
        decoder=decoder,
        num_denoising=4,
        open_vocab=make_ov_args(**overrides.pop("ov_overrides", {})),
    )
    args_kwargs.update(overrides)
    return DETRObjectDetectionTrain(
        DETRObjectDetectionTrainArgs(**VARIANT_ARGS, **args_kwargs)
    )


def make_baseline_train_model(**overrides) -> DETRObjectDetectionTrain:
    torch.manual_seed(0)
    args_kwargs = dict(num_classes=3, decoder="dfine", num_denoising=4)
    args_kwargs.update(overrides)
    return DETRObjectDetectionTrain(
        DETRObjectDetectionTrainArgs(**VARIANT_ARGS, **args_kwargs)
    )


def smoke_batch():
    return {
        "image": torch.rand(2, 3, 224, 224),
        "bboxes": [
            torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
        ],
        "classes": [torch.tensor([1]), torch.tensor([0])],
        "original_size": [(320, 320), (320, 320)],
    }


# ── §8.1 / G0: off-by-default bit-for-bit ───────────────────────────────


def test_baseline_has_no_open_vocab_surface():
    model = make_baseline_train_model().model
    assert not any("ov_" in key for key in model.state_dict())
    assert getattr(model.decoder, "_ov_text_embed_dim", None) is None


def test_baseline_construction_is_reproducible():
    """Two baseline builds under the same seed agree bitwise — the anchor
    for every parity claim in this module (doc 27 G0)."""
    first = make_baseline_train_model()
    second = make_baseline_train_model()
    assert first.model.state_dict().keys() == second.model.state_dict().keys()
    first.model.eval()
    second.model.eval()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        out_a = first.model(x)
        out_b = second.model(x)
    torch.testing.assert_close(out_a[0], out_b[0])
    torch.testing.assert_close(out_a[1], out_b[1])


def test_open_vocab_construction_is_reproducible():
    first = make_train_model()
    second = make_train_model()
    assert first.model.state_dict().keys() == second.model.state_dict().keys()
    first.model.eval()
    second.model.eval()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        out_a = first.model(x)
        out_b = second.model(x)
    torch.testing.assert_close(out_a[0], out_b[0])


# ── §8.2/§8.3: the contrastive head ─────────────────────────────────────


def test_contrastive_head_reduced_view_contract():
    head = ContrastiveClassHead(in_dim=8, num_classes=3, text_dim=16)
    features = torch.randn(2, 5, 8)
    embeds = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    logits = head(features, embeds)
    assert logits.shape == (2, 5, 3)
    # Deterministic: same inputs, same logits.
    torch.testing.assert_close(head(features, embeds), logits)
    # Value identity: logits == scale * cos + bias, computed by hand.
    manual = (
        head.logit_scale.clamp(max=head.logit_scale_max)
        * torch.nn.functional.normalize(head.proj(features), dim=-1)
        @ embeds.T
        + head.class_bias
    )
    torch.testing.assert_close(logits, manual)


def test_contrastive_head_calibration_matches_baseline_prior():
    head = ContrastiveClassHead(in_dim=8, num_classes=3, text_dim=16)
    assert head.class_bias is not None
    assert float(head.class_bias[0].detach()) == pytest.approx(bias_init_with_prob(0.01))
    assert class_bias_init() == pytest.approx(bias_init_with_prob(0.01))
    head.reset_calibration(-2.0)
    assert float(head.class_bias.mean().detach()) == pytest.approx(-2.0)


def test_contrastive_head_scale_clamps_but_still_learns():
    head = ContrastiveClassHead(in_dim=8, num_classes=3, text_dim=16)
    features = torch.randn(4, 8)
    embeds = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
    with torch.no_grad():
        head.logit_scale.fill_(100.0)
    clamped = head(features, embeds)
    with torch.no_grad():
        head.logit_scale.fill_(head.logit_scale_max)
    at_max = head(features, embeds)
    torch.testing.assert_close(clamped, at_max)
    # Below the ceiling the scale carries gradient (CLIP-style learning).
    with torch.no_grad():
        head.logit_scale.fill_(2.0)
    head(features, embeds).sum().backward()
    assert head.logit_scale.grad is not None
    assert float(head.logit_scale.grad.abs()) > 0


def test_head_without_class_bias():
    head = ContrastiveClassHead(in_dim=8, num_classes=3, text_dim=16, use_class_bias=False)
    assert head.class_bias is None
    logits = head(torch.randn(2, 8), torch.nn.functional.normalize(torch.randn(3, 16), dim=-1))
    assert logits.shape == (2, 3)


# ── §8.4: the reduced view keeps the loss stack unchanged ───────────────


def test_loss_key_sets_match_baseline():
    baseline = make_baseline_train_model()
    open_vocab = make_train_model()
    batch = smoke_batch()
    baseline_loss = baseline.training_step(batch, step=0).log_dict
    ov_loss = open_vocab.training_step(batch, step=0).log_dict
    diagnostic = {"ov_logit_scale"}
    # The ONLY allowed difference is the open-vocab diagnostic: every
    # baseline loss key survives unchanged (the reduced-view red line,
    # doc 27 §7).
    assert set(ov_loss) - set(baseline_loss) == diagnostic
    assert torch.isfinite(ov_loss["loss"])
    # The similarity scale rides the log_dict in log-space.
    head = open_vocab.model.decoder.dec_score_head[0]
    assert float(ov_loss["ov_logit_scale"]) == float(head.logit_scale.detach())


def test_validation_step_and_metrics_run():
    wrapper = make_train_model(ov_overrides={"novel_eval_classes": ("bus",)})
    wrapper.set_categories({0: "car", 1: "person", 2: "bus"})
    batch = smoke_batch()
    wrapper.training_step(batch, step=0)
    result = wrapper.validation_step(batch, step=0)
    assert torch.isfinite(result.loss)
    metrics = wrapper.compute_metrics("val")
    assert "val_metric/base_ap" in metrics
    assert "val_metric/novel_ap" in metrics


# ── §8.5: the zero-init text attention is an exact identity ─────────────


def test_text_attention_zero_gate_is_bitwise_identity():
    torch.manual_seed(0)
    sublayer = TextCrossAttention(d_model=16, text_dim=8, num_heads=4)
    target = torch.randn(2, 7, 16)
    embeds = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    assert float(sublayer.gate.detach()) == 0.0
    with torch.no_grad():
        out = sublayer(target, embeds)
    torch.testing.assert_close(out, target)  # bitwise at the zero gate
    with torch.no_grad():
        sublayer.gate.fill_(1.0)
    moved = sublayer(target, embeds)
    assert not torch.allclose(moved, target)


def test_decoder_layers_gain_text_attention_only_when_enabled():
    plain = RTDETRv2Decoder(num_classes=3, hidden_dim=32, num_queries=4, num_layers=2)
    attn = RTDETRv2Decoder(
        num_classes=3,
        hidden_dim=32,
        num_queries=4,
        num_layers=2,
        text_embed_dim=8,
        decoder_text_attention=True,
    )
    for layer in plain.decoder.layers:
        assert layer.text_attn is None
    for layer in attn.decoder.layers:
        assert layer.text_attn is not None
    # v1-only: no layer sublayer without the flag even with a text head.
    head_only = RTDETRv2Decoder(
        num_classes=3, hidden_dim=32, num_queries=4, num_layers=2, text_embed_dim=8
    )
    for layer in head_only.decoder.layers:
        assert layer.text_attn is None


# ── §8.6: v2 language-guided selection structure + Dome smoke ───────────


def test_language_guided_selection_swaps_the_enc_head():
    model = make_task_model(language_guided_selection=True)
    decoder = model.decoder
    assert decoder._ov_lgs is True
    assert isinstance(decoder.enc_score_head, ContrastiveClassHead)
    assert all(isinstance(h, ContrastiveClassHead) for h in decoder.dec_score_head)
    # Baseline v1: the enc head stays the closed-set Linear.
    v1_model = make_task_model()
    assert not isinstance(v1_model.decoder.enc_score_head, ContrastiveClassHead)


def test_dome_dynamic_selection_with_text_head_smoke():
    """R3 mitigation: the Dome dynamic path (per-class NMS over the
    reduced view) runs end to end with the open-vocab heads on."""
    torch.manual_seed(0)
    args = DETRObjectDetectionTrainArgs(
        num_classes=3,
        backbone="dinov2-vittest-notpretrained",
        image_size=(224, 224),
        num_queries=12,
        num_top_queries=12,
        decoder="dfine",
        num_denoising=4,
        density=DETRDensityArgs(max_num_select=20, mwas_window_size=5),
        open_vocab=make_ov_args(language_guided_selection=True),
    )
    wrapper = DETRObjectDetectionTrain(args)
    result = wrapper.training_step(smoke_batch(), step=0)
    assert torch.isfinite(result.loss)


# ── §8.7: the vocabulary cache ──────────────────────────────────────────


def test_identity_embedder_is_deterministic_and_normalized():
    embeds_a = encode_class_names(VOCAB, text_encoder="identity")
    embeds_b = encode_class_names(VOCAB, text_encoder="identity")
    torch.testing.assert_close(embeds_a, embeds_b)
    assert embeds_a.shape == (3, resolve_text_embed_dim("identity"))
    torch.testing.assert_close(
        embeds_a.norm(dim=-1), torch.ones(3), check_dtype=False
    )


def test_cache_roundtrip_and_miss(tmp_path, monkeypatch):
    calls = {"tower": 0}

    def fake_tower(*args, **kwargs):  # pragma: no cover - must never run
        calls["tower"] += 1
        raise AssertionError("the text tower ran on a cache hit")

    cache_dir = tmp_path / "cache"
    # Pre-seed the cache with a known tensor (a real first call would
    # download the tower — out of scope for the hermetic suite):
    cache_file = text_cache_path("clip-b32", VOCAB, "a photo of a {}.", None, cache_dir)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    seeded = torch.nn.functional.normalize(torch.randn(3, 512), dim=-1)
    torch.save(seeded, cache_file)
    monkeypatch.setattr(ov, "_encode_with_tower", fake_tower)
    loaded = encode_class_names(
        VOCAB, text_encoder="clip-b32", cache_dir=cache_dir
    )
    torch.testing.assert_close(loaded, seeded)
    assert calls["tower"] == 0
    # A different vocabulary lands on a different cache key.
    other = text_cache_path(
        "clip-b32", ("car",), "a photo of a {}.", None, cache_dir
    )
    assert other != cache_file
    # A shape-corrupt cache is detected by shape and re-encodes — the
    # (patched) tower then refuses, so the loud path is observable.
    torch.save(torch.zeros(1, 1), cache_file)
    with pytest.raises(AssertionError):
        encode_class_names(VOCAB, text_encoder="clip-b32", cache_dir=cache_dir)


def test_identity_cache_file_written(tmp_path):
    cache_dir = tmp_path / "cache"
    encode_class_names(VOCAB, text_encoder="identity", cache_dir=cache_dir)
    cache_file = text_cache_path("identity", VOCAB, "a photo of a {}.", None, cache_dir)
    assert cache_file.exists()
    restored = torch.load(cache_file, weights_only=True)
    assert restored.shape == (3, resolve_text_embed_dim("identity"))


# ── §8.8: state-dict / export / optimizer surface ───────────────────────


def test_buffer_ships_and_tower_stays_out():
    model = make_task_model()
    keys = list(model.state_dict())
    assert any(key.endswith("ov_text_embeds") for key in keys)
    assert not any("text_model" in key or "vision_model" in key for key in keys)
    # The buffer carries the normalized identity embeddings, not zeros.
    buffer = model.decoder.ov_text_embeds
    assert buffer.norm(dim=-1).sub(1.0).abs().max() < 1e-5


def test_logit_scale_lands_in_the_no_weight_decay_bucket():
    model = make_task_model()
    wd, no_wd = get_weight_decay_parameters([model.decoder])
    no_wd_ids = {id(p) for p in no_wd}
    scales = [h.logit_scale for h in model.decoder.dec_score_head]
    biases = [h.class_bias for h in model.decoder.dec_score_head]
    assert all(id(p) in no_wd_ids for p in scales)
    assert all(id(p) in no_wd_ids for p in biases)
    # The projections are regular parameters (weight decay applies).
    proj_wd = {id(p) for p in wd}
    assert all(id(h.proj.weight) in proj_wd for h in model.decoder.dec_score_head)


def test_load_checkpoint_buffer_tolerance(tmp_path):
    """Checkpoint tolerance for the open-vocab buffer (doc 27 §3.9).

    - A present buffer wins: the loaded checkpoint value lands on the
      model (resume fidelity).
    - An absent buffer (a pre-buffer artifact) is filled from the values
      the constructor just encoded — the strict load survives.
    - A RAW baseline (Linear-head) state stays LOUD: contrastive-head
      weights cannot come from a Linear-head checkpoint; the warm-start
      workflow is the coco_checkpoint loader (head skipping), not a raw
      strict load.
    """
    source = make_train_model()
    own_embeds = source.model.decoder.ov_text_embeds.clone()
    state = {key: value.clone() for key, value in source.state_dict().items()}
    buffer_keys = [key for key in state if key.endswith("ov_text_embeds")]
    assert buffer_keys
    for key in buffer_keys:
        state[key] = torch.randn_like(state[key])  # checkpoint wins

    target = make_train_model()
    target.load_state_dict(state, strict=True)
    torch.testing.assert_close(
        target.model.decoder.ov_text_embeds, state[buffer_keys[0]]
    )

    without_buffer = {
        key: value for key, value in state.items()
        if not key.endswith("ov_text_embeds")
    }
    target = make_train_model()
    target.load_state_dict(without_buffer, strict=True)
    torch.testing.assert_close(
        target.model.decoder.ov_text_embeds, own_embeds
    )

    baseline = make_baseline_train_model()
    baseline_state = baseline.state_dict()
    with pytest.raises(RuntimeError):
        make_train_model().load_state_dict(baseline_state, strict=True)


# ── §8.9: fail-loud paths and config validators ─────────────────────────


def test_unfilled_buffer_forward_raises():
    torch.manual_seed(0)
    decoder = RTDETRv2Decoder(
        num_classes=3, hidden_dim=32, num_queries=4, num_layers=1, text_embed_dim=8
    )
    with pytest.raises(ValueError, match="never loaded"):
        decoder._ov_forward_text_embeds()
    with pytest.raises(ValueError, match="shape"):
        decoder.load_ov_text_embeds(torch.zeros(4, 8))
    decoder.load_ov_text_embeds(torch.randn(3, 8))
    assert decoder._ov_forward_text_embeds() is not None


def test_train_args_reject_vocabulary_count_mismatch():
    with pytest.raises(ValueError, match="num_classes"):
        DETRObjectDetectionTrainArgs(
            **VARIANT_ARGS,
            num_classes=2,
            decoder="dfine",
            open_vocab=make_ov_args(),  # 3 names vs num_classes=2
        )


def test_open_vocab_args_validators():
    with pytest.raises(ValueError, match="unique"):
        make_ov_args(class_names=("car", "car", "bus"))
    with pytest.raises(ValueError, match="slot"):
        make_ov_args(prompt_template="a photo of a class")
    with pytest.raises(ValueError, match="not all in class_names"):
        make_ov_args(novel_eval_classes=("truck",))
    with pytest.raises(ValueError, match="non-empty"):
        make_ov_args(novel_eval_classes=())
    with pytest.raises(ValueError, match="non-empty"):
        make_ov_args(class_names=())
    with pytest.raises(ValueError, match="logit_scale_max"):
        make_ov_args(logit_scale_init=5.0, logit_scale_max=4.6)


def test_novel_eval_full_set_is_the_g4_self_check():
    """The FULL vocabulary as the novel set is legal (doc 27 §3.7 G4):
    novel_ap degrades to the plain per-class-AP mean (the protocol's
    reproduction check) and the empty base subset reports the -1 sentinel."""
    args = make_ov_args(novel_eval_classes=VOCAB)
    assert args.novel_eval_classes == VOCAB
    from autovision.tasks.detection.detr.train_model import (
        DETRObjectDetectionTrain as Train,
    )

    rows = [
        {"name": "car", "ap": 0.5},
        {"name": "person", "ap": 0.7},
        {"name": "bus", "ap": 0.1},
    ]
    result = Train._base_novel_ap(_StubEvaluator(rows), args)
    assert result["novel_ap"] == pytest.approx((0.5 + 0.7 + 0.1) / 3)
    assert result["base_ap"] == -1.0


def test_set_categories_cross_check():
    wrapper = make_train_model()
    wrapper.set_categories({0: "car", 1: "person", 2: "bus"})  # ok
    with pytest.raises(ValueError, match="categories"):
        wrapper.set_categories({0: "car", 1: "person", 2: "truck"})


# ── §8.10: the leave-out protocol aggregates ────────────────────────────


class _StubEvaluator:
    def __init__(self, rows):
        self.last_per_class = rows


def test_base_novel_ap_aggregation():
    from autovision.tasks.detection.detr.train_model import (
        DETRObjectDetectionTrain as Train,
    )

    args = make_ov_args(novel_eval_classes=("bus",))
    rows = [
        {"name": "car", "ap": 0.5},
        {"name": "person", "ap": 0.7},
        {"name": "bus", "ap": 0.1},
        # A class without GT is absent; a -1.0 ap is skipped.
        {"name": "ghost", "ap": -1.0},
    ]
    result = Train._base_novel_ap(_StubEvaluator(rows), args)
    assert result["novel_ap"] == pytest.approx(0.1)
    assert result["base_ap"] == pytest.approx(0.6)
    # Only base classes carry GT → the novel subset aggregates to -1.
    empty = Train._base_novel_ap(_StubEvaluator([]), args)
    assert empty["novel_ap"] == -1.0
    assert empty["base_ap"] == -1.0


# ── §8.11: decoder-family matrix ────────────────────────────────────────


@pytest.mark.parametrize("decoder", ["rtdetrv2", "dfine", "dfine_official"])
@pytest.mark.parametrize("text_attention", [False, True])
def test_decoder_matrix_train_and_eval(decoder: str, text_attention: bool):
    model = make_task_model(
        decoder=decoder,
        language_guided_selection=True,
        decoder_text_attention=text_attention,
    )
    model.eval()
    x = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        logits, boxes = model(x)
    assert logits.shape[-1] == 3
    assert boxes.shape[-1] == 4
    model.train()
    targets = [
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]), "labels": torch.tensor([1])},
        {
            "boxes": torch.tensor([[0.3, 0.3, 0.1, 0.1]]),
            "labels": torch.tensor([0]),
        },
    ]
    with torch.no_grad():
        out = model._forward_train(torch.rand(2, 3, 224, 224), targets=targets)
    assert out["pred_logits"].shape == (2, 12, 3)
    assert torch.isfinite(out["pred_logits"]).all()


def test_warmstart_decoder_key_contract():
    """The key-set contract behind the coco_checkpoint skip fix (doc 27
    §4): under the open-vocab branch, checkpoint score-head tensors vanish
    (renamed proj/logit_scale/class_bias) and the fresh own-side tensors
    are exactly the cls-dependent heads plus the text buffer — the sets
    the warm-start loader skips/tolerates by design."""
    baseline_dec = RTDETRv2Decoder(
        num_classes=80, hidden_dim=32, num_queries=4, num_layers=1
    )
    ov_dec = RTDETRv2Decoder(
        num_classes=3,
        hidden_dim=32,
        num_queries=4,
        num_layers=1,
        text_embed_dim=8,
    )
    ov_dec.load_ov_text_embeds(torch.randn(3, 8))
    baseline_keys = set(baseline_dec.state_dict())
    own_keys = set(ov_dec.state_dict())
    cls_dependent = ("dec_score_head", "enc_score_head", "denoising_class_embed")
    vanished = baseline_keys - own_keys
    assert vanished
    assert all(key.startswith(cls_dependent) for key in vanished)
    fresh = own_keys - baseline_keys
    assert fresh
    assert all(
        key.startswith(("dec_score_head", "enc_score_head"))
        or key.endswith("ov_text_embeds")
        for key in fresh
    )
    # With the text cross-attention enabled, the fresh set grows by the
    # zero-init sublayers — exactly the ".text_attn." keys the warm-start
    # loader tolerates (doc 27 review P1-1).
    attn_dec = RTDETRv2Decoder(
        num_classes=3,
        hidden_dim=32,
        num_queries=4,
        num_layers=1,
        text_embed_dim=8,
        decoder_text_attention=True,
    )
    attn_keys = set(attn_dec.state_dict())
    attn_fresh = attn_keys - baseline_keys
    assert any(".text_attn." in key for key in attn_fresh)
    assert all(
        key.startswith(("dec_score_head", "enc_score_head"))
        or ".text_attn." in key
        or key.endswith("ov_text_embeds")
        for key in attn_fresh
    )


def test_optimizer_names_the_ov_calibration_group():
    """Doc 26 W1 / doc 27 §3.9 discipline: the calibration scalars get an
    explicitly named no-weight-decay group at the base LR, and they do not
    ride the detector groups."""
    wrapper = make_train_model(ov_overrides={"language_guided_selection": True})
    optimizer, _ = wrapper.get_optimizer(
        total_steps=100, global_batch_size=2, train_num_batches=10
    )
    by_name = {group["name"]: group for group in optimizer.param_groups}
    assert "ov_no_wd" in by_name
    ov_group_ids = {id(p) for p in by_name["ov_no_wd"]["params"]}
    assert by_name["ov_no_wd"]["weight_decay"] == 0.0
    scales = {id(h.logit_scale) for h in wrapper.model.decoder.dec_score_head}
    assert scales <= ov_group_ids
    enc_head = wrapper.model.decoder.enc_score_head
    assert id(enc_head.logit_scale) in ov_group_ids
    for name in ("detector", "detector_no_wd"):
        assert by_name[name]["params"] and not (
            {id(p) for p in by_name[name]["params"]} & ov_group_ids
        )


def test_official_wide_tail_heads_are_contrastive():
    """layer_scale > 1 splits the official score heads into a narrow head
    block (up to eval_idx) and a scaled wide tail — the contrastive heads
    must follow the per-layer widths (doc 27 §8.11 / risk R5). Built
    directly: layer_scale is an official-decoder ctor knob the task model
    keeps at its default."""
    from autovision.tasks.detection.shared.dfine_official_decoder import (
        DFINEOfficialDecoder,
    )

    torch.manual_seed(0)
    decoder = DFINEOfficialDecoder(
        num_classes=3,
        hidden_dim=32,
        num_queries=6,
        feat_channels=[32, 32, 32],
        num_levels=3,
        num_layers=4,
        eval_idx=1,
        layer_scale=2,
        text_embed_dim=8,
    )
    decoder.load_ov_text_embeds(torch.randn(3, 8))
    heads = decoder.dec_score_head
    assert all(isinstance(h, ContrastiveClassHead) for h in heads)
    assert [h.proj.in_features for h in heads[:2]] == [32, 32]
    assert [h.proj.in_features for h in heads[2:]] == [64, 64]


def test_official_decoder_deploy_with_text_head():
    """P0 regression (doc 27 review): after ``convert_to_deploy`` the
    non-eval score heads are ``nn.Identity`` — the dispatch must call them
    feature-only (the unconditional layer-0 ``pre_scores`` call runs in
    eval too)."""
    from autovision.tasks.detection.shared.dfine_official_decoder import (
        DFINEOfficialDecoder,
    )

    torch.manual_seed(0)
    decoder = DFINEOfficialDecoder(
        num_classes=3,
        hidden_dim=32,
        num_queries=6,
        feat_channels=[32, 32, 32],
        num_levels=3,
        num_layers=3,
        text_embed_dim=8,
    )
    decoder.load_ov_text_embeds(torch.randn(3, 8))
    decoder.convert_to_deploy()
    decoder.eval()
    feats = [
        torch.rand(1, 32, 28, 28),
        torch.rand(1, 32, 14, 14),
        torch.rand(1, 32, 7, 7),
    ]
    with torch.no_grad():
        out = decoder(feats)
    assert out["pred_logits"].shape == (1, 6, 3)
    assert torch.isfinite(out["pred_logits"]).all()


# ── §8.12: args round trip ──────────────────────────────────────────────


def test_init_args_snapshot_round_trip():
    model = make_task_model()
    snapshot = model._init_args["open_vocab"]
    assert isinstance(snapshot, dict)
    restored = DETROpenVocabArgs(**snapshot)
    assert restored.class_names == VOCAB
    assert restored.text_encoder == "identity"
    assert restored.language_guided_selection is False


def test_export_state_dict_carries_embeds():
    wrapper = make_train_model()
    exported = wrapper.get_export_state_dict()
    assert any(key.endswith("ov_text_embeds") for key in exported)
