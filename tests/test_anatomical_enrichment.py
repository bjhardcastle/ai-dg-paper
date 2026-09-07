"""Arithmetic and interface contracts for discovery anatomy analysis."""

import numpy as np
import polars as pl
import pytest

import dg.anatomical_enrichment


def _unit_qc_fixture() -> pl.DataFrame:
    rows = []
    row_index = 0
    for mouse_index in range(5):
        for session_index in range(2):
            source = f"fixture_mouse_{mouse_index}_session_{session_index}.nwb"
            for local_index in range(30):
                region = "VISp" if local_index < 10 else "LGd" if local_index < 20 else "fiber"
                magnitude = 3.0 if region == "VISp" else 1.0 if region == "LGd" else 0.5
                rows.append(
                    {
                        "_nwb_path": source,
                        "_table_index": local_index,
                        "id": row_index + 100,
                        "peak_channel_id": row_index + 1_000,
                        "subject_id": f"fixture_mouse_{mouse_index}",
                        "ecephys_session_id": 10_000 + 2 * mouse_index + session_index,
                        "well_isolated": True,
                        "axis_weight": magnitude * (-1.0 if local_index % 2 else 1.0),
                        "unit_inclusion_status": "included_isolation_only",
                        "engaged_block_consistency_status": "not_applied_fixture",
                        "fixture_region": region,
                    }
                )
                row_index += 1
    return pl.DataFrame(rows).with_columns(pl.col("_table_index").cast(pl.UInt32))


def _anatomy_inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    units = _unit_qc_fixture()
    labels = units.select(
        "_nwb_path",
        "_table_index",
        pl.col("fixture_region").alias("structure_layer"),
    )
    electrodes = units.select(
        "_nwb_path",
        pl.col("peak_channel_id").alias("id"),
    ).with_columns(
        pl.col("id").cast(pl.Int64).alias("probe_id"),
        pl.col("id").cast(pl.Float64).alias("x"),
        (pl.col("id") * 0.5).cast(pl.Float64).alias("y"),
        (pl.col("id") * 0.25).cast(pl.Float64).alias("z"),
    )
    ontology = pl.DataFrame(
        {
            "acronym": ["VISp", "LGd", "fiber"],
            "structure_id": [385, 170, 9000],
            "structure_name": ["Primary visual area", "LGd", "fixture fiber tract"],
            "major_division_id": [315, 549, None],
            "major_division_acronym": ["Isocortex", "TH", None],
            "major_division_name": ["Isocortex", "Thalamus", None],
        },
        schema_overrides={
            "structure_id": pl.Int64,
            "major_division_id": pl.Int64,
        },
    )
    return units, labels, electrodes, ontology


def _assembled() -> pl.DataFrame:
    return dg.anatomical_enrichment.assemble_unit_anatomy(*_anatomy_inputs())


def test_assemble_unit_anatomy_preserves_all_units_and_maps_parent_regions() -> None:
    assembled = _assembled()

    assert assembled.height == 300
    assert assembled.select("_nwb_path", "_table_index").n_unique() == 300
    assert assembled.filter(pl.col("anatomy_analysis_included")).height == 200
    assert set(
        assembled.filter(pl.col("anatomy_analysis_included")).get_column("parent_region_acronym")
    ) == {"Isocortex", "TH"}
    excluded = assembled.filter(pl.col("structure_acronym") == "fiber")
    assert set(excluded.get_column("anatomy_inclusion_status")) == {
        "excluded_outside_locked_parent_level"
    }
    assert assembled.get_column("spike_arrays_intentionally_not_loaded").all()
    assert not assembled.get_column("confirmation_accessed").any()


def test_assemble_rejects_incomplete_structure_label_rows() -> None:
    units, labels, electrodes, ontology = _anatomy_inputs()

    with pytest.raises(ValueError, match="cover every unit"):
        dg.anatomical_enrichment.assemble_unit_anatomy(
            units,
            labels.head(labels.height - 1),
            electrodes,
            ontology,
        )


def test_analysis_uses_locked_coverage_and_session_then_mouse_aggregation() -> None:
    result = dg.anatomical_enrichment.analyze_anatomical_enrichment(
        _assembled(),
        seed=7,
        n_bootstrap=200,
        n_permutations=200,
        code_version="fixture",
    )

    coverage = result.region_coverage.filter(pl.col("coverage_eligible"))
    assert set(coverage.get_column("parent_region_acronym")) == {"Isocortex", "TH"}
    assert set(coverage.get_column("n_mice")) == {5}
    assert set(coverage.get_column("n_sessions")) == {10}
    assert set(coverage.get_column("median_units_per_represented_mouse")) == {20.0}
    assert result.session_effects.height == 40
    assert result.mouse_effects.height == 20
    assert result.statistics.height == 4
    assert result.permutation_null.height == 800
    assert result.statistics.get_column("adjusted_p_value").is_not_null().all()
    assert set(result.statistics.get_column("analysis_tier")) == {"discovery"}

    isocortex = result.statistics.filter(pl.col("result_id") == "absolute_loading__Isocortex").row(
        0, named=True
    )
    thalamus = result.statistics.filter(pl.col("result_id") == "absolute_loading__TH").row(
        0, named=True
    )
    assert isocortex["estimate"] > 0
    assert thalamus["estimate"] < 0
    assert isocortex["n_mice"] == 5
    assert isocortex["n_sessions"] == 10


def test_session_standardization_has_zero_mean_and_unit_variance() -> None:
    result = dg.anatomical_enrichment.analyze_anatomical_enrichment(
        _assembled(),
        seed=2,
        n_bootstrap=20,
        n_permutations=20,
    )
    check = result.unit_scores.group_by("_nwb_path").agg(
        pl.col("signed_loading_z").mean().alias("signed_mean"),
        pl.col("signed_loading_z").std(ddof=0).alias("signed_sd"),
        pl.col("absolute_loading_z").mean().alias("absolute_mean"),
        pl.col("absolute_loading_z").std(ddof=0).alias("absolute_sd"),
    )

    assert np.max(np.abs(check.get_column("signed_mean").to_numpy())) < 1e-12
    assert np.max(np.abs(check.get_column("absolute_mean").to_numpy())) < 1e-12
    assert np.allclose(check.get_column("signed_sd"), 1.0)
    assert np.allclose(check.get_column("absolute_sd"), 1.0)


def test_permutation_null_is_deterministic_and_annotations_nominate_only_supported_rows() -> None:
    first = dg.anatomical_enrichment.analyze_anatomical_enrichment(
        _assembled(),
        seed=11,
        n_bootstrap=100,
        n_permutations=100,
    )
    second = dg.anatomical_enrichment.analyze_anatomical_enrichment(
        _assembled(),
        seed=11,
        n_bootstrap=100,
        n_permutations=100,
    )

    assert first.permutation_null.equals(second.permutation_null)
    annotated = dg.anatomical_enrichment.add_screen_annotations(
        first.statistics,
        first.region_coverage,
    )
    assert annotated.filter(pl.col("screen_lead")).height == 1
    assert (
        annotated.filter(pl.col("nominated_for_confirmation"))
        .filter(
            (pl.col("score_type") != "absolute_loading")
            | (pl.col("estimate") <= 0)
            | (pl.col("ci_low") <= 0)
            | (pl.col("adjusted_p_value") >= 0.05)
        )
        .is_empty()
    )
