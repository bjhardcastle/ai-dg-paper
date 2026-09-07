"""Contract tests for the discovery-free D05 ontology mapping.

All small tables and payloads below are explicitly arithmetic/interface
fixtures. They are not synthetic neuroscience results and are never written to
the repository results tree.
"""

import hashlib
import pathlib

import polars as pl
import pytest

import dg.ontology


def _ontology_contract_fixture() -> dg.ontology.MajorDivisionOntology:
    division_rows = [
        {
            "major_division_order": index,
            "major_division_id": identifier,
            "major_division_acronym": acronym,
            "major_division_name": name,
        }
        for index, (identifier, acronym, name) in enumerate(
            dg.ontology.EXPECTED_MAJOR_DIVISIONS,
            start=1,
        )
    ]
    structures = []
    for identifier, acronym, name in dg.ontology.EXPECTED_MAJOR_DIVISIONS:
        structures.append(
            {
                "structure_id": identifier,
                "acronym": acronym,
                "structure_name": name,
                "major_division_id": identifier,
                "major_division_acronym": acronym,
                "major_division_name": name,
                "major_division_mapping_status": "mapped_by_structure_id_ancestry",
            }
        )
    structures.append(
        {
            "structure_id": 99_001,
            "acronym": "fixture_hypothalamic_child",
            "structure_name": "Fixture hypothalamic child",
            "major_division_id": 1097,
            "major_division_acronym": "HY",
            "major_division_name": "Hypothalamus",
            "major_division_mapping_status": "mapped_by_structure_id_ancestry",
        }
    )
    return dg.ontology.MajorDivisionOntology(
        structures=pl.DataFrame(structures),
        divisions=pl.DataFrame(division_rows),
        validation=pl.DataFrame({"check": ["fixture"], "status": ["pass"]}),
    )


def _coverage_contract_fixture(*, child_count: int = 700) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "subject_id": ["fixture_mouse_1"] * 6,
            "_nwb_path": ["fixture_session_1.nwb"] * 5 + ["fixture_session_2.nwb"],
            "structure_acronym": [
                "HY",
                "fixture_hypothalamic_child",
                "MB",
                "PAL",
                "STR",
                dg.ontology.MISSING_STRUCTURE_ACRONYM,
            ],
            "n_units": [1, child_count, 2, 3, 4, 5],
            "anatomy_scope": ["raw_allen_acronym_no_parent_mapping"] * 6,
            "analysis_scope": ["coverage_only_no_neural_activity"] * 6,
            # These unused values assert that isolation metrics are not inputs.
            "n_well_isolated_units": [0, child_count, 0, 0, 0, 0],
        }
    )


def test_pinned_source_contract_matches_frozen_author_and_allen_records() -> None:
    assert dg.ontology.AUTHOR_REPOSITORY_COMMIT == ("3c09a0dc2972381f06393ddf5c000e16462c7037")
    assert dg.ontology.AUTHOR_TREE_RESOURCE.size_bytes == 224_537
    assert dg.ontology.AUTHOR_TREE_RESOURCE.sha256 == (
        "eb7df8325ab71f52159f3f2c2720b206a934d3fb10a51edb7e362d2d37ec4ab7"
    )
    assert dg.ontology.STRUCTURE_GRAPH_ID == 1
    assert dg.ontology.STRUCTURE_GRAPH_RESOURCE.size_bytes == 637_735
    assert dg.ontology.STRUCTURE_GRAPH_RESOURCE.sha256 == (
        "b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386"
    )
    assert dg.ontology.MAJOR_DIVISION_STRUCTURE_SET_ID == 687_527_670
    assert dg.ontology.STRUCTURE_SET_RESOURCE.size_bytes == 5_421
    assert dg.ontology.STRUCTURE_SET_RESOURCE.sha256 == (
        "cbb4cda7eb90cd886d9498d6e85de42d241a105cd576fd4255b5e15f921b2aba"
    )
    assert len(dg.ontology.EXPECTED_MAJOR_DIVISIONS) == 12
    assert {row[1] for row in dg.ontology.EXPECTED_MAJOR_DIVISIONS}.issuperset(
        {"HY", "MB", "PAL", "STR"}
    )


def test_cache_is_content_pinned_read_only_and_rejects_changed_content(
    tmp_path: pathlib.Path,
) -> None:
    payload = b"labeled ontology cache fixture\n"
    resource = dg.ontology.PinnedResource(
        key="fixture",
        filename="fixture.txt",
        url="https://example.invalid/fixture.txt",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        authority="test_fixture",
    )
    destination = tmp_path / resource.filename
    calls = []

    status = dg.ontology._cache_pinned_resource(
        resource,
        destination,
        downloader=lambda url: calls.append(url) or payload,
    )
    assert status == "downloaded_and_verified"
    assert calls == [resource.url]
    assert destination.read_bytes() == payload
    assert destination.stat().st_mode & 0o222 == 0

    status = dg.ontology._cache_pinned_resource(
        resource,
        destination,
        downloader=lambda _: pytest.fail("verified cache should not download"),
    )
    assert status == "verified_existing"

    destination.chmod(0o644)
    destination.write_bytes(b"changed ontology cache fixture\n")
    with pytest.raises(RuntimeError, match="size differs|SHA-256 differs"):
        dg.ontology._cache_pinned_resource(
            resource,
            destination,
            downloader=lambda _: payload,
        )


def test_mapping_uses_raw_all_unit_presence_and_ignores_count_magnitude() -> None:
    ontology = _ontology_contract_fixture()
    small = dg.ontology.map_session_anatomy_to_major_divisions(
        _coverage_contract_fixture(child_count=1),
        ontology,
    )
    large = dg.ontology.map_session_anatomy_to_major_divisions(
        _coverage_contract_fixture(child_count=1_000_000),
        ontology,
    )

    assert small.session_presence.equals(large.session_presence)
    assert small.acronym_diagnostics.equals(large.acronym_diagnostics)
    assert small.candidate_diagnostics.equals(large.candidate_diagnostics)
    assert small.session_presence.height == 4
    assert small.session_presence.get_column("has_any_raw_located_unit_in_division").all()
    assert set(small.session_presence.get_column("major_division_acronym")) == {
        "HY",
        "MB",
        "PAL",
        "STR",
    }
    assert small.candidate_diagnostics.get_column("observed_in_m0_coverage").all()
    assert set(small.candidate_diagnostics.get_column("mapping_status")) == {
        "pass_exact_id_ancestry_and_set_membership"
    }
    assert "n_units" not in small.session_presence.columns
    assert "n_well_isolated_units" not in small.session_presence.columns


def test_mapping_rejects_unknown_raw_acronym_and_noncoverage_scope() -> None:
    ontology = _ontology_contract_fixture()
    coverage = _coverage_contract_fixture().with_columns(
        pl.when(pl.col("structure_acronym") == "HY")
        .then(pl.lit("not_in_pinned_tree"))
        .otherwise(pl.col("structure_acronym"))
        .alias("structure_acronym")
    )
    with pytest.raises(RuntimeError, match="absent from the pinned author tree"):
        dg.ontology.map_session_anatomy_to_major_divisions(coverage, ontology)

    invalid_scope = _coverage_contract_fixture().with_columns(
        pl.lit("fixture_neural_activity_scope").alias("analysis_scope")
    )
    with pytest.raises(ValueError, match="unexpected analysis scope"):
        dg.ontology.map_session_anatomy_to_major_divisions(invalid_scope, ontology)
