"""Focused contracts and rendering tests for neural-transition Figure 2."""

import argparse
import datetime
import hashlib
import importlib.util
import json
import pathlib
import sys

import polars as pl
import pytest

import dg.figure_neural_transitions

SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "scripts" / "07_plot_neural_transitions.py"
SPEC = importlib.util.spec_from_file_location("dg_plot_neural_transitions_script", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
PLOT_NEURAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLOT_NEURAL
SPEC.loader.exec_module(PLOT_NEURAL)


def _trajectories() -> pl.DataFrame:
    rows = []
    for mouse_index, subject_id in enumerate(("1", "2", "3")):
        for transition_index, transition in enumerate(("withdrawal", "restoration")):
            for anchor_index, anchor_type in enumerate(("real", "pseudo")):
                for bin_center in (-5.0, -3.0, -1.0, 1.0, 3.0, 5.0):
                    rows.append(
                        {
                            "subject_id": subject_id,
                            "transition": transition,
                            "anchor_type": anchor_type,
                            "bin_center_minutes": bin_center,
                            "mean_axis_score": (
                                0.1 * mouse_index
                                + 0.2 * transition_index
                                - 0.15 * anchor_index
                                + 0.02 * bin_center
                            ),
                            "n_sessions": 2,
                            "n_trials": 12,
                        }
                    )
    return pl.DataFrame(rows)


def _mouse_effects() -> pl.DataFrame:
    rows = []
    values = {
        "withdrawal": ((0.4, 0.1), (0.5, 0.1), (0.4, 0.2)),
        "restoration": ((0.3, 0.1), (0.4, 0.1), (0.5, 0.1)),
    }
    for transition, pairs in values.items():
        for subject_id, (real_effect, pseudo_effect) in zip(("1", "2", "3"), pairs, strict=True):
            rows.append(
                {
                    "subject_id": subject_id,
                    "transition": transition,
                    "real_effect": real_effect,
                    "pseudo_effect": pseudo_effect,
                    "real_minus_pseudo": real_effect - pseudo_effect,
                    "n_sessions": 2,
                }
            )
    return pl.DataFrame(rows)


def _statistics(*, adjusted_name: str = "adjusted_p_value") -> pl.DataFrame:
    rows = []
    effects = _mouse_effects()
    for transition, statistic_id in dg.figure_neural_transitions.PRIMARY_STATISTIC_IDS.items():
        estimate = (
            effects.filter(pl.col("transition") == transition)
            .get_column("real_minus_pseudo")
            .mean()
        )
        rows.append(
            {
                "analysis_id": "neural_transition_axis",
                "result_id": statistic_id,
                "estimate": estimate,
                "ci_low": estimate - 0.1,
                "ci_high": estimate + 0.1,
                "confidence_level": 0.95,
                "p_value": 0.02,
                adjusted_name: 0.04,
                "adjustment_method": "holm",
                "multiplicity_family": "neural_transition_primary",
                "n_mice": 3,
                "n_sessions": 6,
                "status": "pass",
            }
        )
    return pl.DataFrame(rows)


def test_prepare_uses_equal_mouse_weighting_and_complete_bins() -> None:
    prepared = dg.figure_neural_transitions.prepare_neural_transition_figure_data(
        _trajectories(),
        _mouse_effects(),
        _statistics(),
    )

    assert prepared.mouse_trajectories.height == 72
    assert prepared.trajectory_summary.height == 24
    row = prepared.trajectory_summary.filter(
        (pl.col("transition") == "withdrawal")
        & (pl.col("anchor_type") == "real")
        & (pl.col("bin_center_minutes") == -5.0)
    ).row(0, named=True)
    assert row["mean_axis_score"] == pytest.approx(0.0)
    assert row["n_mice"] == 3
    assert row["n_sessions"] == 6
    assert row["n_trials"] == 36
    assert row["ci_low"] <= row["mean_axis_score"] <= row["ci_high"]


def test_prepare_accepts_documented_p_adjusted_alias() -> None:
    prepared = dg.figure_neural_transitions.prepare_neural_transition_figure_data(
        _trajectories(),
        _mouse_effects(),
        _statistics(adjusted_name="p_adjusted"),
    )

    assert "adjusted_p_value" in prepared.statistics.columns
    assert prepared.statistics.get_column("adjusted_p_value").to_list() == [0.04, 0.04]


def test_prepare_rejects_duplicate_mouse_trajectory_row() -> None:
    trajectories = pl.concat((_trajectories(), _trajectories().head(1)))

    with pytest.raises(ValueError, match="duplicate rows"):
        dg.figure_neural_transitions.prepare_neural_transition_figure_data(
            trajectories,
            _mouse_effects(),
            _statistics(),
        )


def test_prepare_rejects_statistic_that_disagrees_with_mouse_effects() -> None:
    statistics = _statistics().with_columns(
        pl.when(pl.col("result_id").str.starts_with("reward_withdrawal"))
        .then(pl.col("estimate") + 0.01)
        .otherwise(pl.col("estimate"))
        .alias("estimate")
    )

    with pytest.raises(ValueError, match="disagrees with the plotted mouse mean"):
        dg.figure_neural_transitions.prepare_neural_transition_figure_data(
            _trajectories(),
            _mouse_effects(),
            statistics,
        )


def test_create_figure_has_two_panel_labels_and_three_axes() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prepared = dg.figure_neural_transitions.prepare_neural_transition_figure_data(
        _trajectories(),
        _mouse_effects(),
        _statistics(),
    )
    figure = PLOT_NEURAL.create_figure(prepared)
    try:
        assert len(figure.axes) == 3
        labels = {text.get_text() for axis in figure.axes for text in axis.texts}
        assert {"A", "B"}.issubset(labels)
    finally:
        plt.close(figure)


def test_analysis_manifest_binds_figure_inputs(tmp_path: pathlib.Path) -> None:
    inputs = {
        key: tmp_path / pathlib.Path(relative).name
        for key, relative in PLOT_NEURAL.INPUT_KEYS.items()
    }
    outputs = {}
    for key, path in inputs.items():
        path.write_text(key, encoding="utf-8")
        outputs[key] = {
            "path": f"tables/{path.name}",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = tmp_path / "neural_transition_analysis_run.json"
    manifest.write_text(
        json.dumps(
            {
                "analysis_id": "neural_transition_axis",
                "run_status": "complete",
                "authoritative": True,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )

    assert PLOT_NEURAL._validated_analysis_manifest(manifest, inputs)["run_status"] == "complete"
    inputs["neural_transition_trajectory"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after"):
        PLOT_NEURAL._validated_analysis_manifest(manifest, inputs)


def test_run_figure_writes_three_formats_and_source_data(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("matplotlib")
    results_root = tmp_path / "results"
    table_root = results_root / "tables"
    table_root.mkdir(parents=True)
    frames = {
        "neural_transition_trajectory": _trajectories(),
        "mouse_neural_transition_effects": _mouse_effects(),
        "neural_transition_statistics": _statistics(),
    }
    outputs = {}
    for key, frame in frames.items():
        destination = results_root / PLOT_NEURAL.INPUT_KEYS[key]
        frame.write_csv(destination)
        outputs[key] = {
            "path": str(destination.relative_to(results_root)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "size_bytes": destination.stat().st_size,
        }
    analysis_manifest = results_root / "manifests" / "neural_transition_analysis_run.json"
    analysis_manifest.parent.mkdir(parents=True)
    analysis_manifest.write_text(
        json.dumps(
            {
                "analysis_id": "neural_transition_axis",
                "analysis_tier": "discovery",
                "run_status": "complete",
                "authoritative": True,
                "outputs": outputs,
            }
        ),
        encoding="utf-8",
    )
    figure_root = results_root / "figures" / "main"
    figure_root.mkdir(parents=True)
    figure_manifest = figure_root / "figure_2_neural_transitions_manifest.json"

    PLOT_NEURAL._run_figure(
        argparse.Namespace(dpi=120),
        started_at=datetime.datetime.now(datetime.UTC),
        results_root=results_root,
        figure_root=figure_root,
        figure_manifest_path=figure_manifest,
    )

    for suffix in ("svg", "pdf", "png"):
        output = figure_root / f"figure_2_neural_transitions.{suffix}"
        assert output.stat().st_size > 1_000
    source_root = figure_root / "figure_2_neural_transitions_source_data"
    assert (source_root / "manifest.json").is_file()
    assert len(list(source_root.glob("*.csv"))) == 4
    assert json.loads(figure_manifest.read_text())["run_status"] == "complete"


@pytest.mark.parametrize(
    ("value", "expected"),
    ((0.02, "0.020"), (0.00099, "<0.001")),
)
def test_adjusted_p_format(value: float, expected: str) -> None:
    assert PLOT_NEURAL._format_adjusted_p(value) == expected
