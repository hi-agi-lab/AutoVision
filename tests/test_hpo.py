# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""HPO (roadmap B2): space spec, random/grid search, ASHA, report artifacts."""

import json
from pathlib import Path

import pytest

from autovision.errors import AutoVisionError
from autovision.hpo import (
    SearchSpace,
    load_space,
    run_search,
    split_overrides,
    task_trial,
)


class TestSpace:
    def test_kinds_and_validation(self):
        space = SearchSpace.from_mapping(
            {
                "model_args.lr": {"loguniform": [1e-5, 1e-3]},
                "model_args.weight_decay": {"choice": [0.0, 1e-4]},
                "model_args.ema_warmup_steps": {"randint": [100, 1000]},
                "model_args.drop_path": {"uniform": [0.0, 0.3]},
                "batch_size": [2, 4, 8],
            }
        )
        assert len(space) == 5
        point = space.sample_many(1, seed=0)[0]
        assert set(point) == {
            "model_args.lr", "model_args.weight_decay",
            "model_args.ema_warmup_steps", "model_args.drop_path", "batch_size",
        }
        assert 1e-5 <= point["model_args.lr"] <= 1e-3
        assert point["batch_size"] in (2, 4, 8)
        assert isinstance(point["model_args.ema_warmup_steps"], int)

    @pytest.mark.parametrize(
        "mapping, match",
        [
            ({"a": {"choice": []}}, "non-empty"),
            ({"a": {"nope": [1, 2]}}, "unknown kind"),
            ({"a": {"uniform": [1.0, 0.5]}}, "low <= high"),
            ({"a": {"loguniform": [0.0, 1.0]}}, "positive lower bound"),
            ({"a": {"randint": [10, 1]}}, "low .* > high"),
            ({"a": {"choice": [1], "uniform": [0, 1]}}, "exactly one kind"),
            ({"": [1]}, "non-empty strings"),
        ],
    )
    def test_invalid_specs_fail_loud(self, mapping, match):
        with pytest.raises(AutoVisionError, match=match):
            SearchSpace.from_mapping(mapping)

    @pytest.mark.parametrize(
        "mapping, match",
        [
            ({"a": {"uniform": [float("nan"), 1.0]}}, "finite"),
            ({"a": {"uniform": [0.0, float("inf")]}}, "finite"),
            ({"a": {"uniform": [1.0]}}, "exactly"),
            ({"a": {"randint": [1.5, 2.5]}}, "whole numbers"),
            ({"a..b": [1]}, "empty segments"),
            ({"a.": [1]}, "empty segments"),
        ],
    )
    def test_more_invalid_specs_fail_loud(self, mapping, match):
        with pytest.raises(AutoVisionError, match=match):
            SearchSpace.from_mapping(mapping)

    def test_grid_deduplicates_repeated_choices(self):
        space = SearchSpace.from_mapping({"a": {"choice": [1, 1, 2]}, "b": [7]})
        assert len(space.grid()) == 2

    def test_dotted_prefix_collision_fails_loud(self):
        with pytest.raises(AutoVisionError, match="dotted prefix"):
            split_overrides({"a.b": 1, "a.b.c": 2})
        with pytest.raises(AutoVisionError, match="dotted prefix"):
            split_overrides({"a.b.c": 2, "a.b": 1})

    def test_seeded_sampling_is_deterministic(self):
        space = SearchSpace.from_mapping({"a": {"uniform": [0.0, 1.0]}})
        assert space.sample_many(5, seed=7) == space.sample_many(5, seed=7)
        assert space.sample_many(5, seed=7) != space.sample_many(5, seed=8)

    def test_grid_product_and_continuous_rejection(self):
        space = SearchSpace.from_mapping({"a": [1, 2], "b": [3, 4, 5]})
        grid = space.grid()
        assert len(grid) == 6
        assert {"a": 1, "b": 3} in grid and {"a": 2, "b": 5} in grid
        with pytest.raises(AutoVisionError, match="continuous axes"):
            SearchSpace.from_mapping({"a": [1, 2], "b": {"uniform": [0, 1]}}).grid()

    def test_load_space_yaml_and_json(self, tmp_path):
        yaml_path = tmp_path / "space.yaml"
        yaml_path.write_text("batch_size: [2, 4]\nmodel_args.lr: {loguniform: [1.0e-5, 1.0e-3]}\n")
        assert len(load_space(yaml_path)) == 2
        json_path = tmp_path / "space.json"
        json_path.write_text(json.dumps({"a": [1, 2]}))
        assert len(load_space(json_path)) == 1
        with pytest.raises(AutoVisionError, match="not found"):
            load_space(tmp_path / "missing.yaml")


class TestSplitOverrides:
    def test_nesting_and_bare_keys(self):
        assert split_overrides(
            {"model_args.lr": 1e-4, "batch_size": 4, "transform_args.mosaic_prob": None}
        ) == {"model_args": {"lr": 1e-4}, "batch_size": 4, "transform_args": {"mosaic_prob": None}}

    def test_scalar_vs_nested_collisions_fail_loud(self):
        # A scalar key and a dotted key on the same path are a dotted-prefix
        # collision, rejected in either insertion order.
        for params in ({"model_args": 1, "model_args.lr": 2},
                       {"model_args.lr": 2, "model_args": 1}):
            with pytest.raises(AutoVisionError, match="dotted prefix"):
                split_overrides(params)


def _synthetic_trial(record_file: Path | None = None):
    """Objective = -(lr - 1e-4)**2 scaled; deterministic and smooth."""

    def trial(params, steps, run_dir):
        if record_file is not None:
            with record_file.open("a") as handle:
                handle.write(f"{run_dir.name} steps={steps}\n")
        lr = float(params["model_args.lr"])
        return {"val_metric/map_50": 0.5 - abs(lr - 1.0e-4) * 1000.0 + steps * 1.0e-6}

    return trial


class TestRandomSearch:
    def _space(self):
        return SearchSpace.from_mapping(
            {"model_args.lr": {"uniform": [1e-5, 5e-4]}, "batch_size": [2, 4]}
        )

    def test_report_best_and_jsonl(self, tmp_path):
        result = run_search(
            _synthetic_trial(), self._space(), tmp_path / "search",
            strategy="random", trials=6, budget=3, objective="val_metric/map_50", seed=1,
        )
        assert result.report_md.exists() and result.trials_jsonl.exists() and result.best_json.exists()
        assert len(result.trials) == 6
        assert all(trial.status == "ok" for trial in result.trials)
        best = result.best
        assert best is not None
        # The synthetic optimum is lr = 1e-4; the best sampled point must be
        # the closest of the six.
        lrs = [float(trial.params["model_args.lr"]) for trial in result.trials]
        assert abs(float(best.params["model_args.lr"]) - 1e-4) == min(
            abs(lr - 1e-4) for lr in lrs
        )
        report = result.report_md.read_text()
        assert "最优配置" in report and "试验明细" in report
        assert f"#{best.index}" in report
        lines = [json.loads(line) for line in result.trials_jsonl.read_text().splitlines()]
        assert [line["index"] for line in lines] == list(range(6))
        assert json.loads(result.best_json.read_text())["index"] == best.index

    def test_minimize_direction(self, tmp_path):
        def trial(params, steps, run_dir):
            return {"val_loss": float(params["batch_size"])}

        result = run_search(
            trial, SearchSpace.from_mapping({"batch_size": [2, 4, 8]}), tmp_path / "min",
            strategy="random", trials=3, budget=1, objective="val_loss", maximize=False, seed=0,
        )
        assert result.best.params["batch_size"] == 2

    def test_failed_trial_is_recorded_and_search_continues(self, tmp_path):
        def trial(params, steps, run_dir):
            if params["batch_size"] == 4:
                raise RuntimeError("boom")
            return {"val_metric/map_50": 0.1}

        result = run_search(
            trial, SearchSpace.from_mapping({"batch_size": [2, 4]}), tmp_path / "fail",
            strategy="random", trials=4, budget=1, objective="val_metric/map_50", seed=0,
        )
        statuses = [trial.status for trial in result.trials]
        assert "failed" in statuses and "ok" in statuses
        assert result.best is not None and result.best.status == "ok"
        assert "boom" in result.report_md.read_text()

    def test_completed_trials_are_reused(self, tmp_path):
        calls = {"count": 0}

        def trial(params, steps, run_dir):
            calls["count"] += 1
            return {"val_metric/map_50": 0.2}

        space = self._space()
        first = run_search(trial, space, tmp_path / "reuse", trials=3, budget=2, strategy="random")
        after_first = calls["count"]
        second = run_search(trial, space, tmp_path / "reuse", trials=3, budget=2, strategy="random")
        assert calls["count"] == after_first  # nothing re-run
        assert all(record.status == "reused" for record in second.trials)
        assert [r.params for r in first.trials] == [r.params for r in second.trials]
        # Every metrics file is backed by a matching provenance sidecar.
        for index in range(3):
            meta_path = tmp_path / "reuse" / f"trial_{index:03d}" / "metrics.json.meta.json"
            meta = json.loads(meta_path.read_text())
            assert meta["schema_version"] == 1
            assert meta["params"] == first.trials[index].params
            assert meta["steps"] == 2
            assert meta["fingerprint"]

    def test_changed_params_or_budget_rerun_instead_of_reusing(self, tmp_path):
        calls = {"count": 0}

        def trial(params, steps, run_dir):
            calls["count"] += 1
            return {"val_metric/map_50": float(params["x"]) * 10.0 + steps}

        out_dir = tmp_path / "changed"
        first = run_search(
            trial, SearchSpace.from_mapping({"x": [1]}), out_dir,
            strategy="grid", trials=1, budget=3, objective="val_metric/map_50",
        )
        assert first.trials[0].status == "ok"
        assert first.trials[0].metrics["val_metric/map_50"] == 13.0

        # Same out_dir, changed space (x=1 -> x=2): the old metrics belong
        # to another trial and MUST be re-run, never reported as "reused".
        second = run_search(
            trial, SearchSpace.from_mapping({"x": [2]}), out_dir,
            strategy="grid", trials=1, budget=3, objective="val_metric/map_50",
        )
        assert calls["count"] == 2
        assert second.trials[0].status == "ok"
        assert second.trials[0].metrics["val_metric/map_50"] == 23.0
        assert json.loads((out_dir / "trial_000" / "metrics.json").read_text()) == {
            "val_metric/map_50": 23.0
        }
        meta_path = out_dir / "trial_000" / "metrics.json.meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["schema_version"] == 1
        assert meta["params"] == {"x": 2}
        assert meta["steps"] == 3
        assert meta["fingerprint"]

        # A changed budget alone is also a different reuse key.
        third = run_search(
            trial, SearchSpace.from_mapping({"x": [2]}), out_dir,
            strategy="grid", trials=1, budget=4, objective="val_metric/map_50",
        )
        assert calls["count"] == 3
        assert third.trials[0].status == "ok"
        assert third.trials[0].metrics["val_metric/map_50"] == 24.0
        assert json.loads(meta_path.read_text())["steps"] == 4

    def test_missing_sidecar_reruns_and_overwrites_legacy_metrics(self, tmp_path):
        calls = {"count": 0}

        def trial(params, steps, run_dir):
            calls["count"] += 1
            return {"val_metric/map_50": float(calls["count"])}

        space = SearchSpace.from_mapping({"x": [1]})
        out_dir = tmp_path / "legacy"
        run_search(trial, space, out_dir, strategy="grid", trials=1, budget=1)
        meta_path = out_dir / "trial_000" / "metrics.json.meta.json"
        assert meta_path.exists()
        meta_path.unlink()  # simulate a run directory written before sidecars

        result = run_search(trial, space, out_dir, strategy="grid", trials=1, budget=1)
        assert calls["count"] == 2
        assert result.trials[0].status == "ok"  # re-run, not reused
        assert result.trials[0].metrics["val_metric/map_50"] == 2.0
        assert json.loads((out_dir / "trial_000" / "metrics.json").read_text()) == {
            "val_metric/map_50": 2.0
        }
        assert meta_path.exists()

    def test_failed_rerun_leaves_no_usable_sidecar(self, tmp_path):
        state = {"fail": False, "value": 1.0}

        def trial(params, steps, run_dir):
            if state["fail"]:
                raise RuntimeError("boom")
            return {"val_metric/map_50": state["value"]}

        out_dir = tmp_path / "failed_rerun"
        run_search(trial, SearchSpace.from_mapping({"x": [1]}), out_dir,
                   strategy="grid", trials=1, budget=1)
        meta_path = out_dir / "trial_000" / "metrics.json.meta.json"
        assert meta_path.exists()

        # The changed trial rejects the old cache, then fails: the sidecar
        # must be gone so no later run can treat the stale metrics as valid.
        state["fail"] = True
        failed = run_search(trial, SearchSpace.from_mapping({"x": [2]}), out_dir,
                            strategy="grid", trials=1, budget=1)
        assert failed.trials[0].status == "failed"
        assert not meta_path.exists()
        # The old (x=1) metrics file survives, but without a sidecar it is
        # not reused: the old trial re-runs and overwrites it.
        state.update(fail=False, value=5.0)
        recovered = run_search(trial, SearchSpace.from_mapping({"x": [1]}), out_dir,
                               strategy="grid", trials=1, budget=1)
        assert recovered.trials[0].status == "ok"
        assert recovered.trials[0].metrics["val_metric/map_50"] == 5.0
        assert meta_path.exists()

    def test_corrupt_metrics_file_reruns_instead_of_crashing(self, tmp_path):
        calls = {"count": 0}

        def trial(params, steps, run_dir):
            calls["count"] += 1
            return {"val_metric/map_50": 0.3}

        space = self._space()
        run_search(trial, space, tmp_path / "corrupt", trials=1, budget=1, strategy="random")
        metrics_file = tmp_path / "corrupt" / "trial_000" / "metrics.json"
        metrics_file.write_text('{"val_metric/map_50": ')  # truncated
        result = run_search(trial, space, tmp_path / "corrupt", trials=1, budget=1, strategy="random")
        assert result.trials[0].status == "ok"  # recovered, not reused
        assert calls["count"] == 2
        metrics_file.write_text("null")  # non-mapping payload
        result = run_search(trial, space, tmp_path / "corrupt", trials=1, budget=1, strategy="random")
        assert result.trials[0].status == "ok"
        assert calls["count"] == 3

    def test_grid_strategy(self, tmp_path):
        result = run_search(
            _synthetic_trial(), SearchSpace.from_mapping({"batch_size": [2, 4], "a": [1, 2]}),
            tmp_path / "grid", strategy="grid", trials=4, budget=1, objective="val_metric/map_50",
        )
        assert len(result.trials) == 4
        with pytest.raises(AutoVisionError, match="enumerates 4 points"):
            run_search(
                _synthetic_trial(), SearchSpace.from_mapping({"a": [1, 2], "b": [3, 4]}),
                tmp_path / "grid2", strategy="grid", trials=2, budget=1,
            )

    def test_unknown_strategy_fails_loud(self, tmp_path):
        with pytest.raises(AutoVisionError, match="Unknown strategy"):
            run_search(_synthetic_trial(), self._space(), tmp_path / "x", strategy="bayes")


class TestAsha:
    def test_promotion_rungs_and_continuation(self, tmp_path):
        record_file = tmp_path / "calls.txt"
        result = run_search(
            _synthetic_trial(record_file),
            SearchSpace.from_mapping({"model_args.lr": {"uniform": [1e-5, 5e-4]}}),
            tmp_path / "asha",
            strategy="asha", trials=9, budget=2, asha_max_budget=8, asha_eta=3,
            objective="val_metric/map_50", seed=3,
        )
        by_rung: dict[int, int] = {}
        for trial in result.trials:
            by_rung[trial.rung] = by_rung.get(trial.rung, 0) + 1
        # 9 at the first rung, 3 at the second, 1 at the third (budget 2/6/18->8: the
        # last rung stops at max_budget).
        assert by_rung == {0: 9, 1: 3, 2: 1}
        budgets = {trial.rung: trial.steps for trial in result.trials}
        assert budgets == {0: 2, 1: 6, 2: 8}  # min(eta * 6, max_budget=8)

        # Promoted configs CONTINUE the same run dir (monotone steps per dir).
        steps_by_dir: dict[str, list[int]] = {}
        for line in record_file.read_text().splitlines():
            name, steps = line.split()
            steps_by_dir.setdefault(name, []).append(int(steps.split("=")[1]))
        promoted = {name: steps for name, steps in steps_by_dir.items() if len(steps) > 1}
        assert promoted, "expected at least one promoted configuration"
        for steps in promoted.values():
            assert steps == sorted(steps) and len(set(steps)) == len(steps)

        # Per-rung metrics files + the report chain.
        for rung in by_rung:
            assert (tmp_path / "asha" / "trial_000" / f"metrics.rung{rung}.json").exists()
        report = result.report_md.read_text()
        assert "策略 asha" in report

    def test_changed_budget_ladder_reruns_stale_rungs(self, tmp_path):
        calls: dict[tuple[str, int], int] = {}

        def trial(params, steps, run_dir):
            key = (run_dir.name, steps)
            calls[key] = calls.get(key, 0) + 1
            return {"val_metric/map_50": 10.0 - float(params["x"])}

        space = SearchSpace.from_mapping({"x": [1, 2]})
        out_dir = tmp_path / "asha_budget"
        first = run_search(
            trial, space, out_dir,
            strategy="asha", trials=2, budget=2, asha_max_budget=6, asha_eta=3,
            objective="val_metric/map_50", seed=0,
        )
        # Ladder 2 -> 6: two configs at rung 0, the best one promoted.
        promoted = [record for record in first.trials if record.rung == 1]
        assert len(promoted) == 1
        assert calls[(Path(promoted[0].run_dir).name, 6)] == 1

        # The identical ladder reuses every rung (idempotence preserved).
        run_search(
            trial, space, out_dir,
            strategy="asha", trials=2, budget=2, asha_max_budget=6, asha_eta=3,
            objective="val_metric/map_50", seed=0,
        )
        assert all(count == 1 for count in calls.values())

        # eta 3 -> 2 changes the promoted rung's budget (6 -> 4): its stale
        # rung-1 metrics must re-run even though params are unchanged.
        second = run_search(
            trial, space, out_dir,
            strategy="asha", trials=2, budget=2, asha_max_budget=6, asha_eta=2,
            objective="val_metric/map_50", seed=0,
        )
        assert all(record.status == "reused" for record in second.trials if record.rung == 0)
        second_rung1 = [record for record in second.trials if record.rung == 1]
        assert len(second_rung1) == 1
        assert second_rung1[0].status == "ok"  # re-run, not reused
        rung1_dir = Path(second_rung1[0].run_dir)
        assert calls[(rung1_dir.name, 4)] == 1
        meta = json.loads((rung1_dir / "metrics.rung1.json.meta.json").read_text())
        assert meta["steps"] == 4
        assert meta["fingerprint"]

    def test_promotion_resumes_against_the_real_training_entry(self, tmp_path):
        """Regression (review blocker): a promoted rung resumes the real run.

        The engine rejects ``checkpoint=`` + ``resume_interrupted`` together;
        passing both made every promotion fail silently. This run goes
        through the ACTUAL train_object_detection on coco8 (CPU).
        """
        result = run_search(
            task_trial(
                "object_detection",
                {
                    "data": "coco8",
                    "model": "rtdetr/resnet18",
                    "devices": "cpu",
                    "batch_size": 2,
                    "model_args": {
                        "image_size": (64, 64),
                        "num_queries": 10,
                        "num_top_queries": 10,
                    },
                },
            ),
            SearchSpace.from_mapping({"model_args.lr": {"choice": [1e-4, 5e-4]}}),
            tmp_path / "asha_real",
            strategy="asha",
            trials=2,
            budget=2,
            asha_max_budget=4,
            asha_eta=2,
            objective="val_metric/map_50",
            seed=0,
        )
        by_rung: dict[int, list] = {}
        for trial in result.trials:
            by_rung.setdefault(trial.rung, []).append(trial)
        assert set(by_rung) == {0, 1}, "expected a promotion rung"
        assert all(trial.status == "ok" for trial in by_rung[0])
        promoted = by_rung[1]
        assert len(promoted) == 1
        assert promoted[0].status == "ok", promoted[0].error
        # The promoted rung continued the same run dir (the run holds the
        # extra steps, not a fresh 2-step run).
        assert (tmp_path / "asha_real" / f"trial_{promoted[0].index:03d}"
                / "run" / "checkpoints").exists()

    def test_eta_validation(self, tmp_path):
        with pytest.raises(AutoVisionError, match="asha_eta"):
            run_search(
                _synthetic_trial(), SearchSpace.from_mapping({"a": [1]}),
                tmp_path / "x", strategy="asha", trials=2, budget=1, asha_eta=1,
            )
        with pytest.raises(AutoVisionError, match="min_budget"):
            run_search(
                _synthetic_trial(), SearchSpace.from_mapping({"a": [1]}),
                tmp_path / "y", strategy="asha", trials=2, budget=5, asha_max_budget=2,
            )


class TestTaskTrial:
    def test_task_entry_overrides_and_resume_convention(self, monkeypatch, tmp_path):
        # task_trial resolves the entry from its own module (a static import
        # the layering scanner sees), so patch it THERE.
        import autovision.cli.train_task.object_detection as entry_module

        captured: list[dict] = []

        class _Result:
            steps_run = 5
            best_metrics = {"val_metric/map_50": 0.42}

        def fake_entry(**kwargs):
            captured.append(kwargs)
            return _Result()

        monkeypatch.setattr(entry_module, "train_object_detection", fake_entry)
        trial = task_trial("object_detection", {"data": "coco8", "model": "m", "devices": "cpu"})

        # Fresh run: no checkpoint kwargs.
        run_dir = tmp_path / "trial_000"
        metrics = trial({"model_args.lr": 1e-4, "batch_size": 4}, 10, run_dir)
        assert metrics == {"val_metric/map_50": 0.42, "steps_run": 5.0}
        assert captured[0]["out"] == run_dir / "run"
        assert captured[0]["model_args"] == {"lr": 1e-4} and captured[0]["batch_size"] == 4
        assert captured[0]["steps"] == 10
        assert "checkpoint" not in captured[0]

        # Promoted rung: a non-empty run dir resumes via resume_interrupted
        # ONLY — the engine rejects checkpoint= + resume_interrupted together
        # ("'checkpoint' and 'resume_interrupted' cannot both be set"), which
        # made every ASHA promotion fail before this was fixed.
        checkpoint_dir = run_dir / "run" / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "last.ckpt").write_bytes(b"stub")
        trial({"model_args.lr": 1e-4}, 20, run_dir)
        assert "checkpoint" not in captured[1]
        assert captured[1]["resume_interrupted"] is True
        assert captured[1]["steps"] == 20  # steps is the TOTAL target
        assert captured[1]["out"] == run_dir / "run"

    def test_resumed_run_without_improvement_recovers_metrics(self, monkeypatch, tmp_path):
        """The engine fills best_metrics only on improvement: a resumed rung
        that did not improve must still yield its own validation read from
        the run's metrics.jsonl (else the ASHA ladder stalls silently)."""
        import autovision.cli.train_task.object_detection as entry_module

        class _Result:
            steps_run = 1
            best_metrics: dict = {}  # no improvement this invocation

        monkeypatch.setattr(entry_module, "train_object_detection", lambda **kwargs: _Result())
        trial = task_trial("object_detection", {"data": "coco8"})
        run_dir = tmp_path / "trial_000"
        run_root = run_dir / "run"
        run_root.mkdir(parents=True)
        (run_root / "metrics.jsonl").write_text(
            '{"ts": 1.0, "step": -1, "hyperparams": {"steps": 2}}\n'
            '{"ts": 2.0, "step": 1, "val_metric/map_50": 0.25, "val_loss": 3.5}\n'
        )
        metrics = trial({"model_args.lr": 1e-4}, 2, run_dir)
        assert metrics["val_metric/map_50"] == 0.25
        assert metrics["val_loss"] == 3.5
        assert metrics["steps_run"] == 1.0
        assert "hyperparams" not in metrics and "step" not in metrics and "ts" not in metrics

    def test_unknown_task_fails_loud(self, tmp_path):
        with pytest.raises(AutoVisionError, match="Unknown HPO task"):
            task_trial("not_a_task", {})

    def test_reserved_axis_names_are_rejected(self, tmp_path):
        trial = task_trial("object_detection", {"data": "coco8"})
        for key in ("out", "steps"):
            with pytest.raises(AutoVisionError, match="reserved"):
                trial({key: 1}, 2, tmp_path / f"t_{key}")


class TestHpoCliErrors:
    def test_missing_space_is_a_clear_error(self):
        from typer.testing import CliRunner

        from autovision.cli._app import app

        result = CliRunner().invoke(app, ["hpo", "out=/tmp/never"])
        assert result.exit_code != 0
        assert "space=" in result.output

    def test_unknown_task_fails_before_running(self):
        from typer.testing import CliRunner

        from autovision.cli._app import app

        space = Path("/tmp") / "hpo_cli_space.yaml"
        space.write_text("batch_size: [2, 4]\n")
        result = CliRunner().invoke(
            app, ["hpo", f"space={space}", "task=not_a_task", "trials=1"]
        )
        assert result.exit_code != 0
        # typer.BadParameter prints to stderr (or the runner stashes it).
        combined = result.output + str(result.exception)
        assert "Unknown HPO task" in combined


class TestHpoCli:
    def test_one_random_search_produces_a_report(self, tmp_path):
        """The P1 gate: one random search run → a report (real training)."""
        from typer.testing import CliRunner

        from autovision.cli._app import app

        space = tmp_path / "space.yaml"
        space.write_text("model_args.lr: {choice: [1.0e-4, 5.0e-4]}\n")
        result = CliRunner().invoke(
            app,
            [
                "hpo", f"space={space}", f"out={tmp_path / 'hpo'}",
                "task=object_detection", "data=coco8", "model=rtdetr/resnet18",
                "trials=2", "budget=2", "devices=cpu", "batch_size=2",
                "objective=val_metric/map_50", "seed=0",
            ],
        )
        assert result.exit_code == 0, result.output
        report = (tmp_path / "hpo" / "report.md").read_text()
        assert "超参搜索结果" in report and "试验明细" in report
        assert (tmp_path / "hpo" / "trials.jsonl").exists()
        assert "report:" in result.output
