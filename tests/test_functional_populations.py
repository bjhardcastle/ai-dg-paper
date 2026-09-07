import numpy as np
import polars as pl
import pytest

import dg.functional_populations


def _trial_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "_nwb_path": ["a"] * 6,
            "_table_index": list(range(6)),
            "change_time": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "change_image_name": ["im1_r-1.0", "im2_r-1.0"] * 3,
            "novel_image_id": ["im9"] * 6,
            "physical_image_change": [True] * 6,
            "reward_block": [
                "engaged_1",
                "engaged_1",
                "no_reward",
                "no_reward",
                "engaged_2",
                "engaged_2",
            ],
            "aborted": [False] * 6,
            "auto_rewarded": [False] * 6,
            "response_in_window": [True, False, False, True, True, False],
            "response_latency_from_licks": [0.2, None, None, 0.3, 0.25, None],
            "lick_times_valid": [True] * 6,
        }
    )


def test_select_functional_population_trials_derives_first_lick_and_halves() -> None:
    selected = dg.functional_populations.select_functional_population_trials(_trial_frame())

    assert selected.height == 6
    assert selected.get_column("assignment_half").to_list() == [0, 1, 0, 1, 0, 1]
    assert selected.get_column("first_response_lick_time").to_list() == [
        1.2,
        None,
        None,
        4.3,
        5.25,
        None,
    ]


def test_select_functional_population_trials_rejects_invalid_response_latency() -> None:
    trials = _trial_frame().with_columns(
        pl.when(pl.col("_table_index") == 0)
        .then(None)
        .otherwise(pl.col("response_latency_from_licks"))
        .alias("response_latency_from_licks")
    )
    with pytest.raises(ValueError, match="finite raw-lick"):
        dg.functional_populations.select_functional_population_trials(trials)


def test_running_summary_uses_nearest_valid_sample_and_marks_large_gaps() -> None:
    config = dg.functional_populations.FunctionalPopulationConfig(
        baseline_window=(-0.2, 0.0),
        early_window=(0.05, 0.15),
        late_window=(0.2, 0.6),
        running_max_gap_seconds=0.06,
    )
    result = dg.functional_populations.summarize_running_at_events(
        np.arange(0.0, 2.0, 0.05),
        np.arange(0.0, 2.0, 0.05) * 2,
        [1.0, 3.0],
        config=config,
    )

    assert result["running_baseline"][0] == pytest.approx(1.8)
    assert result["running_early"][0] == pytest.approx(2.2)
    assert result["running_late"][0] == pytest.approx(2.8)
    assert result["running_late_delta"][0] == pytest.approx(1.0)
    assert all(np.isnan(value) for value in result.row(1))


def test_prepare_running_samples_sorts_and_exact_deduplicates_with_qc() -> None:
    prepared = dg.functional_populations.prepare_running_samples(
        [0.0, 0.2, 0.1, 0.2, np.nan],
        [1.0, 4.0, 2.0, 6.0, 100.0],
    )

    np.testing.assert_allclose(prepared.timestamps, [0.0, 0.1, 0.2])
    np.testing.assert_allclose(prepared.speed, [1.0, 2.0, 5.0])
    assert prepared.usable is True
    assert prepared.n_raw_rows == 5
    assert prepared.n_finite_pairs == 4
    assert prepared.n_unique_timestamps == 3
    assert prepared.n_decreasing_adjacent_raw == 1
    assert prepared.n_exact_duplicates_removed == 1
    assert prepared.status == "pass_sorted_exact_deduplicated_finite_pairs"


def test_prepare_running_samples_marks_one_unique_timestamp_unavailable() -> None:
    prepared = dg.functional_populations.prepare_running_samples([1.0, 1.0], [2.0, 4.0])

    assert prepared.usable is False
    np.testing.assert_allclose(prepared.speed, [3.0])
    assert prepared.status == "unavailable_fewer_than_two_unique_finite_timestamps"
    with pytest.raises(ValueError, match="fewer than two unique"):
        dg.functional_populations.summarize_running_at_events([1.0], [2.0], [1.0])


def test_bin_spike_batch_uses_half_open_bins_and_separate_lick_alignment() -> None:
    config = dg.functional_populations.FunctionalPopulationConfig(bin_width_seconds=0.025)
    batch = dg.functional_populations.bin_spike_batch(
        [[0.75, 1.0, 1.024, 1.025, 1.749, 2.0]],
        [1.0, 2.0],
        [1.2, np.nan],
        config=config,
    )

    assert batch.change_counts.shape == (2, 40, 1)
    assert batch.lick_counts.shape == (1, 24, 1)
    assert batch.change_counts[0, 0, 0] == 1
    assert batch.change_counts[0, 10, 0] == 2
    assert batch.change_counts[0, 11, 0] == 1
    assert batch.change_counts[0, -1, 0] == 1
    assert batch.change_counts[1, 10, 0] == 1


def test_variance_stabilized_response_subtracts_duration_scaled_baseline() -> None:
    counts = np.zeros((2, 4, 1), dtype=int)
    counts[:, 0:2, 0] = 1
    counts[:, 2:4, 0] = 3
    response = dg.functional_populations.variance_stabilized_window_response(
        counts,
        np.array([-0.15, -0.05, 0.05, 0.15]),
        (0.0, 0.2),
        (-0.2, 0.0),
        bin_width_seconds=0.1,
    )

    expected = (np.sqrt(6 + 3 / 8) - np.sqrt(2 + 3 / 8)) / np.sqrt(0.2)
    assert response.shape == (2, 1)
    np.testing.assert_allclose(response[:, 0], expected)


def test_censored_response_uses_only_complete_prelick_bins() -> None:
    counts = np.ones((2, 6, 1), dtype=int)
    counts[:, 4:, 0] = 3
    response = dg.functional_populations.censored_variance_stabilized_window_response(
        counts,
        np.array([-0.15, -0.05, 0.05, 0.15, 0.25, 0.35]),
        [0.31, 0.15],
        (0.15, 0.4),
        (-0.2, 0.0),
        bin_width_seconds=0.1,
    )

    expected = np.sqrt(4 + 3 / 8) / np.sqrt(0.2) - np.sqrt(2 + 3 / 8) / np.sqrt(0.2)
    assert response[0, 0] == pytest.approx(expected)
    assert np.isnan(response[1, 0])


def test_model_design_includes_reward_contrasts_and_adjustment_terms() -> None:
    trials = _trial_frame().with_columns(
        pl.Series("running_baseline", [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        pl.Series("running_early", [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
        pl.Series("running_early_delta", [0.5] * 6),
    )
    design, names = dg.functional_populations.build_model_design(
        trials,
        window="early",
        adjusted=True,
    )

    assert design.shape == (6, len(names))
    assert names[:2] == ("reward_reversible", "engaged_drift")
    assert "lick_response" in names
    assert "running_early_delta" in names
    np.testing.assert_allclose(design[:, 0], [0.5, 0.5, -1.0, -1.0, 0.5, 0.5])
    assert np.isfinite(design).all()


def test_contiguous_block_folds_segments_each_state_in_time() -> None:
    blocks = np.repeat(dg.functional_populations.REWARD_BLOCKS, 6)
    times = np.concatenate([np.arange(6), np.arange(10, 16), np.arange(20, 26)])
    folds = dg.functional_populations.contiguous_block_folds(blocks, times, 3)

    np.testing.assert_array_equal(folds, np.tile(np.repeat(np.arange(3), 2), 3))


def test_nested_contiguous_ridge_returns_oof_predictions_and_coefficients() -> None:
    blocks = np.repeat(dg.functional_populations.REWARD_BLOCKS, 12)
    times = np.arange(36, dtype=float)
    reversible = np.where(blocks == "no_reward", -1.0, 0.5)
    drift = np.where(blocks == "engaged_1", -0.5, np.where(blocks == "engaged_2", 0.5, 0.0))
    nuisance = np.sin(times / 4)
    design = np.column_stack([reversible, drift, nuisance])
    outcomes = np.column_stack([2.0 * reversible + 0.2 * nuisance, -1.5 * reversible + 0.3 * drift])

    result = dg.functional_populations.nested_contiguous_ridge(
        design,
        outcomes,
        blocks,
        times,
    )

    assert result.coefficients.shape == (3, 2)
    assert result.oof_predictions.shape == outcomes.shape
    assert np.isfinite(result.oof_predictions).all()
    assert np.all(result.oof_r2 > 0.9)
    assert result.coefficients[0, 0] > 1.8
    assert result.coefficients[0, 1] < -1.3
    assert set(result.selected_lambdas).issubset(
        dg.functional_populations.DEFAULT_FUNCTIONAL_POPULATION_CONFIG.ridge_lambdas
    )


def test_response_archetypes_require_cross_half_class_and_sign_agreement() -> None:
    halves = np.tile([0, 1], 4)
    responded = np.array([True, True, False, False, True, True, False, False])
    early = np.tile([4.0, 0.5, 0.2], (8, 1))
    late = np.tile([0.5, -5.0, 0.3], (8, 1))
    action = np.tile([0.3, 0.2, 6.0], (4, 1))
    classified = dg.functional_populations.classify_response_archetypes(
        early,
        late,
        action,
        halves,
        responded,
    )

    assert classified.get_column("response_class").to_list() == list(
        dg.functional_populations.RESPONSE_CLASSES
    )
    assert classified.get_column("dominant_response_sign").to_list() == [1.0, -1.0, 1.0]
    assert classified.get_column("class_assignment_stable").all()


def test_reversible_contrast_matches_declared_three_block_formula() -> None:
    values = np.array([[2.0, 4.0], [0.0, 5.0], [4.0, 8.0]])
    observed = dg.functional_populations.reversible_contrast(
        values,
        dg.functional_populations.REWARD_BLOCKS,
    )
    np.testing.assert_allclose(observed, [3.0, 1.0])
