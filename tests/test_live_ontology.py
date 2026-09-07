"""Optional read-only validation of the exact pinned D05 ontology endpoints."""

import os

import pytest

import dg.ontology

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("DG_RUN_LIVE_ONTOLOGY_TESTS") != "1",
        reason="set DG_RUN_LIVE_ONTOLOGY_TESTS=1 to verify pinned ontology endpoints",
    ),
]


def test_exact_endpoint_payloads_match_pins_and_cross_validate(tmp_path) -> None:
    """Download, hash, and cross-check the author tree and official Allen graph/set."""

    sources = dg.ontology.cache_d05_ontology_sources(tmp_path)
    ontology = dg.ontology.load_d05_major_division_ontology(sources)

    assert sources.provenance.get_column("sha256").to_list() == [
        resource.sha256 for resource in dg.ontology.PINNED_RESOURCES
    ]
    assert sources.provenance.get_column("size_bytes").to_list() == [
        resource.size_bytes for resource in dg.ontology.PINNED_RESOURCES
    ]
    assert ontology.structures.height == dg.ontology.EXPECTED_STRUCTURE_COUNT
    assert ontology.divisions.height == 12
    assert ontology.validation.get_column("status").unique().to_list() == ["pass"]
