"""Tests for result-table contracts and mouse-level statistical inference.

The small numeric tables below are algorithmic fixtures, not synthetic
neuroscience observations and not estimates from DANDI:001051. Published data
are never inferred from or represented by these values.
"""

import math

import polars as pl
import pytest

import dg.statistics


def _minimum_statistics_row(**updates):
    row = {
        "analysis_id": "behavior_gating",
        "result_id": "figure_1_change_response",
        "contrast_id": "reversible_reward_availability",
        "hypothesis": "Change responses decrease in NR and return in E2.",
        "dandiset_version": "0.260825.2232",
        "code_version": "algorithm-test-fixture",
        "analysis_tier": "exploratory",
        "inclusion_definition": "technically valid sessions",
        "estimate": 0.4,
        "scale": "probability difference",
        "ci_low": 0.2,
        "ci_high": 0.6,
        "confidence_level": 0.95,
        "ci_method": "mouse percentile bootstrap",
        "test_statistic": 0.4,
        "test_method": "two-sided sign flip",
        "p_value": 0.03,
        "multiplicity_family": "behavior_primary",
        "sidedness": "two-sided",
        "n_mice": 3,
        "n_sessions": 4,
        "n_trials": 100,
        "aggregation": "equal sessions within mouse; equal mice",
        "status": "pass",
    }
    row.update(updates)
    return row


def _algorithmic_session_estimates() -> pl.DataFrame:
    """Return arithmetic-only values for verifying aggregation weights."""

    return pl.DataFrame(
        {
            "mouse_id": ["fixture_a", "fixture_a", "fixture_b"],
            "session_id": ["fixture_a1", "fixture_a2", "fixture_b1"],
            "engaged_1": [0.8, 0.6, 0.4],
            "no_reward": [0.2, 0.1, 0.3],
            "engaged_2": [0.6, 0.8, 0.6],
        }
    )


def _algorithmic_mouse_estimates() -> pl.DataFrame:
    """Return arithmetic-only mouse effects for resampling tests."""

    return pl.DataFrame(
        {
            "mouse_id": ["fixture_a", "fixture_b", "fixture_c"],
            "reversible_contrast": [1.0, 2.0, 3.0],
        }
    )


def test_statistics_table_is_normalized_to_the_canonical_schema() -> None:
    raw = pl.DataFrame([_minimum_statistics_row(analysis_id=" behavior_gating ")])

    normalized = dg.statistics.normalize_statistics_table(raw)

    assert normalized.columns == dg.statistics.STATISTICS_SCHEMA.names()
    assert normalized.schema == dg.statistics.STATISTICS_SCHEMA
    assert normalized.get_column("analysis_id").item() == "behavior_gating"
    assert normalized.get_column("seed").null_count() == 1
    assert normalized.get_column("n_units").null_count() == 1
    assert normalized.get_column("adjusted_p_value").null_count() == 1


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"analysis_id": "  "}, "analysis_id"),
        ({"analysis_tier": "training"}, "analysis_tier"),
        ({"p_value": 1.1}, "p_value"),
        ({"ci_low": 0.7}, "ci_low"),
        ({"n_mice": -1}, "n_mice"),
        ({"n_mice": 2.5}, "integer"),
        ({"ci_method": None}, "confidence interval requires"),
        ({"status": "failed", "reason": None}, "requires a reason"),
        ({"status": "null", "estimate": None}, "requires an estimate"),
        ({"adjusted_p_value": 0.05}, "adjustment_method"),
    ],
)
def test_statistics_table_rejects_semantically_invalid_rows(updates, message: str) -> None:
    table = pl.DataFrame([_minimum_statistics_row(**updates)])

    with pytest.raises(ValueError, match=message):
        dg.statistics.normalize_statistics_table(table)


def test_statistics_table_rejects_unknown_and_missing_required_columns() -> None:
    unknown = pl.DataFrame([_minimum_statistics_row(unregistered_field="value")])
    missing = pl.DataFrame([_minimum_statistics_row()]).drop("result_id")

    with pytest.raises(ValueError, match="unknown columns"):
        dg.statistics.normalize_statistics_table(unknown)
    with pytest.raises(ValueError, match="missing required columns"):
        dg.statistics.normalize_statistics_table(missing)


def test_holm_adjustment_preserves_order_nulls_and_monotonicity() -> None:
    adjusted = dg.statistics.holm_adjust([0.01, 0.04, 0.03, 0.002, None])

    assert adjusted[:4] == pytest.approx([0.03, 0.06, 0.06, 0.008])
    assert adjusted[4] is None


def test_holm_adjustment_is_applied_separately_by_declared_family() -> None:
    rows = [
        _minimum_statistics_row(
            result_id="a1",
            multiplicity_family="family_a",
            p_value=0.01,
        ),
        _minimum_statistics_row(
            result_id="a2",
            multiplicity_family="family_a",
            p_value=0.04,
        ),
        _minimum_statistics_row(
            result_id="b1",
            multiplicity_family="family_b",
            p_value=0.04,
        ),
    ]

    adjusted = dg.statistics.add_holm_adjustment(pl.DataFrame(rows))

    assert adjusted.get_column("adjusted_p_value").to_list() == pytest.approx([0.02, 0.04, 0.04])
    assert adjusted.get_column("adjustment_method").to_list() == ["holm"] * 3


def test_reversible_contrast_uses_equal_session_weight_within_mouse() -> None:
    mouse_estimates = dg.statistics.aggregate_reversible_contrast_by_mouse(
        _algorithmic_session_estimates()
    )

    assert mouse_estimates.to_dicts() == [
        {
            "mouse_id": "fixture_a",
            "n_sessions": 2,
            "engaged_1_mean": pytest.approx(0.7),
            "no_reward_mean": pytest.approx(0.15),
            "engaged_2_mean": pytest.approx(0.7),
            "reversible_contrast": pytest.approx(0.55),
        },
        {
            "mouse_id": "fixture_b",
            "n_sessions": 1,
            "engaged_1_mean": pytest.approx(0.4),
            "no_reward_mean": pytest.approx(0.3),
            "engaged_2_mean": pytest.approx(0.6),
            "reversible_contrast": pytest.approx(0.2),
        },
    ]
    assert mouse_estimates.get_column("reversible_contrast").mean() == pytest.approx(0.375)


def test_reversible_contrast_rejects_duplicate_or_incomplete_sessions() -> None:
    sessions = _algorithmic_session_estimates()
    duplicate = pl.concat([sessions, sessions.head(1)])
    incomplete = sessions.with_columns(
        pl.when(pl.col("session_id") == "fixture_a1")
        .then(pl.lit(None))
        .otherwise(pl.col("engaged_1"))
        .alias("engaged_1")
    )

    with pytest.raises(ValueError, match="unique mouse/session"):
        dg.statistics.aggregate_reversible_contrast_by_mouse(duplicate)
    with pytest.raises(ValueError, match="engaged_1"):
        dg.statistics.aggregate_reversible_contrast_by_mouse(incomplete)


def test_mouse_bootstrap_is_seeded_and_weights_each_mouse_once() -> None:
    mice = _algorithmic_mouse_estimates()

    first = dg.statistics.bootstrap_mouse_mean(
        mice,
        seed=721,
        n_resamples=2_000,
        batch_size=137,
    )
    second = dg.statistics.bootstrap_mouse_mean(
        mice,
        seed=721,
        n_resamples=2_000,
        batch_size=137,
    )

    assert first == second
    assert first.estimate == 2.0
    assert first.n_mice == 3
    assert first.ci_low <= first.estimate <= first.ci_high
    assert 1.0 <= first.ci_low <= first.ci_high <= 3.0


def test_sign_flip_test_is_exact_for_small_mouse_samples() -> None:
    mice = pl.DataFrame(
        {
            "mouse_id": ["fixture_a", "fixture_b"],
            "reversible_contrast": [1.0, 2.0],
        }
    )

    result = dg.statistics.two_sided_sign_flip_test(mice, seed=721)

    assert result.statistic == 1.5
    assert result.p_value == 0.5
    assert result.method == "exact_two_sided_sign_flip"
    assert result.n_permutations == 4
    assert result.seed is None


def test_sign_flip_monte_carlo_is_seeded_and_uses_plus_one_correction() -> None:
    mice = _algorithmic_mouse_estimates()

    first = dg.statistics.two_sided_sign_flip_test(
        mice,
        seed=721,
        exact_max_mice=2,
        n_resamples=20_000,
        batch_size=257,
    )
    second = dg.statistics.two_sided_sign_flip_test(
        mice,
        seed=721,
        exact_max_mice=2,
        n_resamples=20_000,
        batch_size=257,
    )

    assert first == second
    assert first.method == "monte_carlo_two_sided_sign_flip"
    assert first.seed == 721
    assert first.p_value == pytest.approx(0.25, abs=0.02)
    assert first.p_value >= 1 / (first.n_permutations + 1)


def test_sign_flip_rejects_an_intractable_exact_enumeration_request() -> None:
    with pytest.raises(ValueError, match="exact_max_mice"):
        dg.statistics.two_sided_sign_flip_test(
            _algorithmic_mouse_estimates(),
            seed=721,
            exact_max_mice=25,
        )


@pytest.mark.parametrize(
    "values",
    [
        [math.nan, 0.1],
        [math.inf, 0.1],
        [None, 0.1],
    ],
)
def test_mouse_inference_rejects_invalid_effects(values) -> None:
    mice = pl.DataFrame(
        {
            "mouse_id": ["fixture_a", "fixture_b"],
            "reversible_contrast": values,
        },
        schema_overrides={"reversible_contrast": pl.Float64},
    )

    with pytest.raises(ValueError, match="finite, non-null"):
        dg.statistics.bootstrap_mouse_mean(mice, seed=721)
