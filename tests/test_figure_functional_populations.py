import importlib.util
import pathlib

import numpy as np
import polars as pl
import pytest

import dg.figure_functional_populations
import dg.functional_populations
import dg.statistics


def _mouse_psths() -> pl.DataFrame:
    config = dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG
    edges = np.arange(
        config.change_start_seconds,
        config.change_stop_seconds + config.bin_width_seconds / 2,
        config.bin_width_seconds,
    )
    centers = (edges[:-1] + edges[1:]) / 2
    rows = []
    for mouse_index, mouse in enumerate(("1", "2", "3")):
        for group_index, group in enumerate(("early_sensory", "late_action")):
            for block_index, block in enumerate(("engaged_1", "no_reward", "engaged_2")):
                for center in centers:
                    rows.append(
                        {
                            "subject_id": mouse,
                            "aligned_event": "change",
                            "response_group": group,
                            "reward_block": block,
                            "bin_center_seconds": center,
                            "mean_sign_aligned_rate_hz": (
                                mouse_index * 0.1 + group_index + block_index * 0.2 + center
                            ),
                            "n_sessions": 2,
                            "n_unit_session_contributions": 20,
                            "n_trial_session_contributions": 100,
                        }
                    )
    return pl.DataFrame(rows)


def _mouse_effects() -> pl.DataFrame:
    rows = []
    for index, mouse in enumerate(("1", "2", "3")):
        early_total = 0.1 + index * 0.02
        late_total = 0.4 + index * 0.03
        early_adjusted = 0.08 + index * 0.02
        late_adjusted = 0.3 + index * 0.025
        early_raw = 0.12 + index * 0.01
        late_raw = 0.35 + index * 0.02
        rows.append(
            {
                "subject_id": mouse,
                "n_sessions": 2,
                "n_units": 100,
                "n_trials": 80,
                "early_raw": early_raw,
                "late_raw": late_raw,
                "late_minus_early_raw": late_raw - early_raw,
                "early_total": early_total,
                "late_total": late_total,
                "late_minus_early_total": late_total - early_total,
                "early_adjusted": early_adjusted,
                "late_adjusted": late_adjusted,
                "late_minus_early_adjusted": late_adjusted - early_adjusted,
                "late_adjustment_change": late_adjusted - late_total,
            }
        )
    return pl.DataFrame(rows)


def _statistics(effects: pl.DataFrame) -> pl.DataFrame:
    mapping = {
        "late_minus_early_model_free": "late_minus_early_raw",
        "late_minus_early_model_total": "late_minus_early_total",
        "late_minus_early_model_adjusted": "late_minus_early_adjusted",
        "late_state_model_total": "late_total",
        "late_state_model_adjusted": "late_adjusted",
    }
    rows = []
    for result_id, value_column in mapping.items():
        estimate = float(effects.get_column(value_column).mean())
        rows.append(
            {
                "analysis_id": "functional_populations",
                "result_id": result_id,
                "contrast_id": value_column,
                "hypothesis": result_id,
                "dandiset_version": "0.260825.2232",
                "code_version": "test",
                "seed": 1,
                "analysis_tier": "discovery",
                "inclusion_definition": "test fixture",
                "missingness_stratum": "complete",
                "estimate": estimate,
                "scale": "test units",
                "ci_low": estimate - 0.1,
                "ci_high": estimate + 0.1,
                "confidence_level": 0.95,
                "ci_method": "fixture",
                "test_statistic": estimate,
                "test_method": "fixture",
                "p_value": 0.02,
                "adjusted_p_value": 0.04,
                "adjustment_method": "holm",
                "multiplicity_family": "figure_3_primary_functional_timing",
                "sidedness": "two-sided",
                "n_mice": effects.height,
                "n_sessions": 6,
                "n_probes": None,
                "n_units": 300,
                "n_trials": 240,
                "aggregation": "mouse",
                "bootstrap_id": "fixture",
                "model_formula": "fixture",
                "cv_grouping": "fixture",
                "status": "pass",
                "reason": None,
            }
        )
    return dg.statistics.normalize_statistics_table(pl.DataFrame(rows))


def _stability() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": ["1", "2", "3"],
            "n_sessions": [2, 2, 2],
            "n_units": [100, 100, 100],
            "n_stable_units": [60, 70, 80],
            "n_early_sensory": [20, 20, 30],
            "n_late_prelick": [20, 25, 25],
            "n_action_related": [20, 25, 25],
            "mean_assignment_margin": [0.4, 0.5, 0.6],
            "stable_assignment_fraction": [0.6, 0.7, 0.8],
        }
    )


def test_prepare_functional_population_figure_data_validates_and_summarizes() -> None:
    effects = _mouse_effects()
    prepared = dg.figure_functional_populations.prepare_functional_population_figure_data(
        _mouse_psths(),
        effects,
        _statistics(effects),
        _stability(),
    )

    assert prepared.mouse_psths.height == 3 * 2 * 3 * 40
    assert prepared.psth_summary.height == 2 * 3 * 40
    assert prepared.statistics.height == 5
    assert prepared.class_stability.height == 3
    assert prepared.psth_summary.get_column("n_mice").unique().to_list() == [3]


def test_prepare_rejects_effect_identity_mismatch() -> None:
    effects = _mouse_effects().with_columns(
        (pl.col("late_minus_early_adjusted") + 0.1).alias("late_minus_early_adjusted")
    )
    with pytest.raises(ValueError, match="does not reconstruct"):
        dg.figure_functional_populations.prepare_functional_population_figure_data(
            _mouse_psths(), effects, _statistics(_mouse_effects()), _stability()
        )


def test_prepare_rejects_incomplete_psth_bins() -> None:
    psths = _mouse_psths().slice(1)
    effects = _mouse_effects()
    with pytest.raises(ValueError, match="complete bins"):
        dg.figure_functional_populations.prepare_functional_population_figure_data(
            psths, effects, _statistics(effects), _stability()
        )


def test_create_figure_has_four_axes() -> None:
    pytest.importorskip("matplotlib")
    script_path = (
        pathlib.Path(__file__).parents[1] / "scripts" / "09_plot_functional_populations.py"
    )
    spec = importlib.util.spec_from_file_location("plot_functional_populations", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    effects = _mouse_effects()
    prepared = dg.figure_functional_populations.prepare_functional_population_figure_data(
        _mouse_psths(), effects, _statistics(effects), _stability()
    )
    figure = module.create_figure(prepared)
    try:
        assert len(figure.axes) == 4
    finally:
        import matplotlib.pyplot as plt

        plt.close(figure)
