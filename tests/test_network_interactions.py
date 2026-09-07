"""Contracts for the discovery-only simultaneous-network analysis."""

import numpy as np
import polars as pl
import pytest

import dg.network_interactions


def test_network_trial_selector_censors_licks_through_target_window() -> None:
    trials = pl.DataFrame(
        {
            "_table_index": list(range(7)),
            "change_time": [10.0] * 7,
            "change_image_name": ["im001_r-1.0"] * 7,
            "novel_image_id": ["im999_r-1.0"] * 7,
            "physical_image_change": [True] * 7,
            "reward_block": ["engaged_1"] * 7,
            "aborted": [False] * 7,
            "auto_rewarded": [False] * 7,
            "lick_times": [
                [],
                [9.85],
                [9.849],
                [10.15],
                [10.30],
                [10.301],
                [float("nan")],
            ],
            "lick_times_valid": [True] * 7,
            "response_in_window": [False] * 7,
        }
    )

    selected = dg.network_interactions.select_network_trials(trials)

    assert selected.get_column("_table_index").to_list() == [0, 2, 5]
    assert selected.get_column("network_lick_censor_start_seconds").unique().item() == -0.15
    assert selected.get_column("network_lick_censor_stop_seconds").unique().item() == 0.30


def test_population_features_use_half_open_positive_lag_windows() -> None:
    config = dg.network_interactions.NetworkConfig(
        minimum_trials_per_block=4,
        minimum_units_per_region=2,
        maximum_units_per_region=2,
        minimum_pair_mice=2,
        minimum_pair_sessions=2,
        minimum_median_units_per_mouse=2,
        outer_folds=2,
        inner_folds=2,
        ranks=(1, 2),
    )
    early, late = dg.network_interactions.build_windowed_population_features(
        [[0.85, 0.999, 1.0, 1.149, 1.15, 1.299, 1.30]],
        [1.0],
        config=config,
    )

    assert early.shape == late.shape == (1, 1)
    assert early.item() == pytest.approx(0.0)
    assert late.item() == pytest.approx(0.0)


def _network_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    generator = np.random.default_rng(1051)
    n_per_block = 30
    n_units = 8
    blocks = np.repeat(dg.network_interactions.REWARD_BLOCKS, n_per_block)
    times = np.arange(blocks.size, dtype=float)
    source = generator.normal(size=(blocks.size, n_units))
    history = generator.normal(scale=0.4, size=(blocks.size, n_units))
    nuisance = np.column_stack((np.linspace(-1, 1, blocks.size), times % 2))
    mapping = np.diag(np.linspace(1.0, 0.3, n_units))
    target = 0.25 * history + generator.normal(scale=0.08, size=history.shape)
    engaged = blocks != dg.network_interactions.NO_REWARD
    target[engaged] += source[engaged] @ mapping
    return source, history, target, nuisance, blocks, times


def test_nested_network_fit_recovers_engaged_positive_lag_and_shift_control() -> None:
    source, history, target, nuisance, blocks, times = _network_fixture()
    result = dg.network_interactions.fit_network_pair_by_block(
        source,
        history,
        target,
        nuisance,
        blocks,
        times,
    )

    assert result.block_performance.height == 3
    assert result.fold_performance.height == 12
    assert result.balanced_trial_indices.size == 90
    lookup = {row["reward_block"]: row for row in result.block_performance.iter_rows(named=True)}
    assert lookup["engaged_1"]["source_added_r2"] > 0.5
    assert lookup["engaged_2"]["source_added_r2"] > 0.5
    assert lookup["no_reward"]["source_added_r2"] < 0.2
    assert lookup["engaged_1"]["real_minus_shift_source_added_r2"] > 0.4
    assert lookup["engaged_2"]["real_minus_shift_source_added_r2"] > 0.4


def test_network_fit_balances_trials_across_blocks() -> None:
    source, history, target, nuisance, blocks, times = _network_fixture()
    keep = np.ones(blocks.size, dtype=bool)
    keep[np.flatnonzero(blocks == "engaged_1")[-4:]] = False
    keep[np.flatnonzero(blocks == "engaged_2")[-2:]] = False

    result = dg.network_interactions.fit_network_pair_by_block(
        source[keep],
        history[keep],
        target[keep],
        nuisance[keep],
        blocks[keep],
        times[keep],
    )

    assert result.block_performance.get_column("n_trials").to_list() == [26, 26, 26]
    selected_blocks = blocks[keep][result.balanced_trial_indices]
    assert {block: int((selected_blocks == block).sum()) for block in set(selected_blocks)} == {
        "engaged_1": 26,
        "no_reward": 26,
        "engaged_2": 26,
    }


def _unit_anatomy() -> pl.DataFrame:
    rows = []
    for mouse_index in range(5):
        for session_index in range(2):
            source = f"s{mouse_index}_{session_index}"
            for region in ("VISp", "MOs"):
                for unit_index in range(10):
                    rows.append(
                        {
                            "_nwb_path": source,
                            "subject_id": f"m{mouse_index}",
                            "network_region": region,
                            "well_isolated": True,
                            "_table_index": unit_index + (100 if region == "MOs" else 0),
                        }
                    )
    return pl.DataFrame(rows)


def test_region_pair_coverage_enforces_mouse_session_and_unit_thresholds() -> None:
    coverage = dg.network_interactions.build_region_pair_coverage(_unit_anatomy())

    assert coverage.height == 2
    assert coverage.get_column("coverage_eligible").all()
    assert set(coverage.get_column("n_mice")) == {5}
    assert set(coverage.get_column("n_sessions")) == {10}


def _session_performance() -> pl.DataFrame:
    rows = []
    for mouse_index in range(5):
        for session_index in range(2):
            for source_region, target_region, multiplier in (
                ("VISp", "MOs", 1.0),
                ("MOs", "VISp", 0.2),
            ):
                for block, real, shift in (
                    ("engaged_1", 0.12, 0.02),
                    ("no_reward", 0.01, 0.02),
                    ("engaged_2", 0.10, 0.02),
                ):
                    rows.append(
                        {
                            "_nwb_path": f"s{mouse_index}_{session_index}",
                            "subject_id": f"m{mouse_index}",
                            "source_region": source_region,
                            "target_region": target_region,
                            "reward_block": block,
                            "source_added_r2": multiplier * real,
                            "shift_source_added_r2": multiplier * shift,
                            "real_minus_shift_source_added_r2": multiplier * (real - shift),
                            "n_trials": 20,
                            "model_status": "pass",
                        }
                    )
    return pl.DataFrame(rows)


def test_mouse_summary_and_nomination_preserve_mouse_as_replicate() -> None:
    mouse_blocks, mouse_effects = dg.network_interactions.summarize_network_by_mouse(
        _session_performance()
    )
    screen = dg.network_interactions.infer_and_nominate_network_pair(
        mouse_effects,
        seed=1051,
        n_bootstrap=200,
        n_sign_flips=200,
    )

    assert mouse_blocks.height == 5 * 2 * 3
    assert mouse_effects.height == 5 * 2
    assert screen.get_column("nominated").sum() == 1
    nominated = screen.filter(pl.col("nominated")).row(0, named=True)
    assert (nominated["source_region"], nominated["target_region"]) == ("VISp", "MOs")
    assert nominated["n_mice"] == 5
    assert nominated["estimate"] == pytest.approx(0.10)
