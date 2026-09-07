"""Pinned Allen CCF sources and outcome-blind D05 major-division mapping."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

import polars as pl

AUTHOR_REPOSITORY_COMMIT = "3c09a0dc2972381f06393ddf5c000e16462c7037"
STRUCTURE_GRAPH_ID = 1
MAJOR_DIVISION_STRUCTURE_SET_ID = 687527670
MISSING_STRUCTURE_ACRONYM = "[missing]"
EXPECTED_STRUCTURE_COUNT = 1327
EXPECTED_CANDIDATE_ACRONYMS = ("HY", "MB", "PAL", "STR")


@dataclasses.dataclass(frozen=True)
class PinnedResource:
    """One immutable remote payload required by the D05 ontology contract."""

    key: str
    filename: str
    url: str
    sha256: str
    size_bytes: int
    authority: str


@dataclasses.dataclass(frozen=True)
class CachedOntologySources:
    """Validated local copies and portable provenance rows."""

    paths: dict[str, pathlib.Path]
    provenance: pl.DataFrame


@dataclasses.dataclass(frozen=True)
class MajorDivisionOntology:
    """Validated author tree with official 12-division membership."""

    structures: pl.DataFrame
    divisions: pl.DataFrame
    validation: pl.DataFrame


@dataclasses.dataclass(frozen=True)
class SessionDivisionMapping:
    """Presence-only mapping artifacts derived from the M0 unit inventory."""

    session_presence: pl.DataFrame
    acronym_diagnostics: pl.DataFrame
    candidate_diagnostics: pl.DataFrame


AUTHOR_TREE_RESOURCE = PinnedResource(
    key="author_ccf_structure_tree_2017",
    filename="ccf_structure_tree_2017.csv",
    url=(
        "https://raw.githubusercontent.com/AllenInstitute/"
        "SHIELD_Dynamic_Gating_Analysis/"
        f"{AUTHOR_REPOSITORY_COMMIT}/analysis_code/ccf_structure_tree_2017.csv"
    ),
    sha256="eb7df8325ab71f52159f3f2c2720b206a934d3fb10a51edb7e362d2d37ec4ab7",
    size_bytes=224_537,
    authority="pinned_author_analysis_repository",
)
STRUCTURE_GRAPH_RESOURCE = PinnedResource(
    key="allen_structure_graph_1",
    filename="allen_structure_graph_1.json",
    url="https://api.brain-map.org/api/v2/structure_graph_download/1.json",
    sha256="b5f0d024d1df09ee18aef15093f55240f4a70b3924ee9689add612dadff80386",
    size_bytes=637_735,
    authority="official_allen_brain_map_api",
)
STRUCTURE_SET_RESOURCE = PinnedResource(
    key="allen_structure_set_687527670",
    filename="allen_structure_set_687527670.json",
    url=(
        "https://api.brain-map.org/api/v2/data/query.json?criteria="
        "model::Structure,rma::criteria,%5Bgraph_id%24eq1%5D,"
        "structure_sets%5Bid%24eq687527670%5D&"
        "rma::options%5Bnum_rows%24eqall%5D"
    ),
    sha256="cbb4cda7eb90cd886d9498d6e85de42d241a105cd576fd4255b5e15f921b2aba",
    size_bytes=5_421,
    authority="official_allen_brain_map_api",
)
PINNED_RESOURCES = (
    AUTHOR_TREE_RESOURCE,
    STRUCTURE_GRAPH_RESOURCE,
    STRUCTURE_SET_RESOURCE,
)

EXPECTED_MAJOR_DIVISIONS = (
    (1097, "HY", "Hypothalamus"),
    (313, "MB", "Midbrain"),
    (698, "OLF", "Olfactory areas"),
    (803, "PAL", "Pallidum"),
    (315, "Isocortex", "Isocortex"),
    (703, "CTXsp", "Cortical subplate"),
    (477, "STR", "Striatum"),
    (354, "MY", "Medulla"),
    (549, "TH", "Thalamus"),
    (512, "CB", "Cerebellum"),
    (1089, "HPF", "Hippocampal formation"),
    (771, "P", "Pons"),
)


def cache_d05_ontology_sources(
    cache_directory: str | os.PathLike[str],
    *,
    timeout_seconds: float = 60.0,
) -> CachedOntologySources:
    """Download missing pinned payloads and reject changed cached content."""

    root = pathlib.Path(cache_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, pathlib.Path] = {}
    rows = []
    for resource in PINNED_RESOURCES:
        path = root / resource.filename
        _cache_pinned_resource(
            resource,
            path,
            downloader=lambda url: _download_bytes(url, timeout_seconds=timeout_seconds),
        )
        paths[resource.key] = path
        rows.append(
            {
                "resource_key": resource.key,
                "authority": resource.authority,
                "source_url": resource.url,
                "repository_commit": (
                    AUTHOR_REPOSITORY_COMMIT if resource.key == AUTHOR_TREE_RESOURCE.key else None
                ),
                "structure_graph_id": (
                    STRUCTURE_GRAPH_ID if resource.key != AUTHOR_TREE_RESOURCE.key else None
                ),
                "structure_set_id": (
                    MAJOR_DIVISION_STRUCTURE_SET_ID
                    if resource.key == STRUCTURE_SET_RESOURCE.key
                    else None
                ),
                "cached_filename": resource.filename,
                "sha256": resource.sha256,
                "size_bytes": resource.size_bytes,
                "cache_status": "verified_pinned",
                "cache_file_mode": "0444_read_only",
                "immutability_policy": "reject_any_sha256_or_size_difference",
            }
        )
    return CachedOntologySources(
        paths=paths,
        provenance=pl.DataFrame(rows, infer_schema_length=None),
    )


def load_d05_major_division_ontology(
    sources: CachedOntologySources,
) -> MajorDivisionOntology:
    """Validate all three sources and build the ancestry-based division map."""

    for resource in PINNED_RESOURCES:
        path = sources.paths.get(resource.key)
        if path is None:
            raise RuntimeError(f"ontology sources lack pinned resource {resource.key}")
        _validate_payload(path.read_bytes(), resource)
    author = _read_author_tree(sources.paths[AUTHOR_TREE_RESOURCE.key])
    official = _read_official_structure_graph(sources.paths[STRUCTURE_GRAPH_RESOURCE.key])
    divisions = _read_official_structure_set(sources.paths[STRUCTURE_SET_RESOURCE.key])
    _validate_author_against_official(author, official)
    _validate_divisions(divisions, author, official)

    division_ids = set(divisions.get_column("major_division_id").to_list())
    division_lookup = {row["major_division_id"]: row for row in divisions.iter_rows(named=True)}
    mapped_rows = []
    for row in author.iter_rows(named=True):
        path_ids = _parse_structure_id_path(row["structure_id_path"])
        matches = [identifier for identifier in path_ids if identifier in division_ids]
        if len(matches) > 1:
            raise RuntimeError(
                f"structure {row['acronym']} descends from multiple major divisions: {matches}"
            )
        division = division_lookup[matches[0]] if matches else None
        mapped_rows.append(
            {
                **row,
                "major_division_id": (
                    division["major_division_id"] if division is not None else None
                ),
                "major_division_acronym": (
                    division["major_division_acronym"] if division is not None else None
                ),
                "major_division_name": (
                    division["major_division_name"] if division is not None else None
                ),
                "major_division_mapping_status": (
                    "mapped_by_structure_id_ancestry"
                    if division is not None
                    else "outside_official_12_major_divisions"
                ),
            }
        )
    structures = pl.DataFrame(mapped_rows, infer_schema_length=None).sort("structure_id")
    _validate_expected_candidates(structures, divisions, official)
    validation = pl.DataFrame(
        {
            "check": [
                "author_payload_sha256",
                "official_graph_payload_sha256",
                "official_set_payload_sha256",
                "author_official_id_set",
                "author_official_parent_links",
                "author_official_structure_paths",
                "official_major_division_membership",
                "expected_candidate_mappings",
            ],
            "status": ["pass"] * 8,
            "detail": [
                AUTHOR_TREE_RESOURCE.sha256,
                STRUCTURE_GRAPH_RESOURCE.sha256,
                STRUCTURE_SET_RESOURCE.sha256,
                f"{EXPECTED_STRUCTURE_COUNT} identical structure IDs",
                f"{EXPECTED_STRUCTURE_COUNT} identical parent links",
                f"{EXPECTED_STRUCTURE_COUNT} identical ancestry paths",
                f"{len(EXPECTED_MAJOR_DIVISIONS)} members in set {MAJOR_DIVISION_STRUCTURE_SET_ID}",
                ", ".join(EXPECTED_CANDIDATE_ACRONYMS),
            ],
        }
    )
    return MajorDivisionOntology(
        structures=structures,
        divisions=divisions,
        validation=validation,
    )


def map_session_anatomy_to_major_divisions(
    unit_session_anatomy_coverage: pl.DataFrame | pl.LazyFrame,
    ontology: MajorDivisionOntology,
) -> SessionDivisionMapping:
    """Map raw session acronyms to division presence using all units as booleans."""

    coverage = (
        unit_session_anatomy_coverage.collect()
        if isinstance(unit_session_anatomy_coverage, pl.LazyFrame)
        else unit_session_anatomy_coverage
    )
    required = {
        "subject_id",
        "_nwb_path",
        "structure_acronym",
        "n_units",
        "anatomy_scope",
        "analysis_scope",
    }
    missing = sorted(required.difference(coverage.columns))
    if missing:
        raise ValueError(f"unit-session anatomy coverage lacks columns: {missing}")
    if coverage.is_empty():
        raise ValueError("unit-session anatomy coverage is empty")
    if coverage.get_column("subject_id").dtype != pl.String:
        raise TypeError("unit-session anatomy coverage subject_id must be String")
    if coverage.get_column("_nwb_path").dtype != pl.String:
        raise TypeError("unit-session anatomy coverage _nwb_path must be String")
    if coverage.get_column("structure_acronym").dtype != pl.String:
        raise TypeError("unit-session anatomy coverage structure_acronym must be String")
    for column in ("subject_id", "_nwb_path", "structure_acronym"):
        if coverage.get_column(column).null_count():
            raise ValueError(f"unit-session anatomy coverage {column} contains nulls")
        if coverage.filter(pl.col(column).str.strip_chars() == "").height:
            raise ValueError(f"unit-session anatomy coverage {column} contains empty values")
    if coverage.filter(pl.col("subject_id") != pl.col("subject_id").str.strip_chars()).height:
        raise ValueError("unit-session anatomy coverage subject_id is not canonical")
    if set(coverage.get_column("anatomy_scope").unique().to_list()) != {
        "raw_allen_acronym_no_parent_mapping"
    }:
        raise ValueError("unit-session anatomy coverage has an unexpected anatomy scope")
    if set(coverage.get_column("analysis_scope").unique().to_list()) != {
        "coverage_only_no_neural_activity"
    }:
        raise ValueError("unit-session anatomy coverage has an unexpected analysis scope")
    if (
        coverage.select("subject_id", "_nwb_path", "structure_acronym").n_unique()
        != coverage.height
    ):
        raise ValueError("unit-session anatomy coverage rows must be unique by session/acronym")
    if not coverage.get_column("n_units").dtype.is_integer():
        raise TypeError("unit-session anatomy coverage n_units must be an integer")
    if (
        coverage.get_column("n_units").null_count()
        or coverage.filter(pl.col("n_units") <= 0).height
    ):
        raise ValueError("unit-session anatomy coverage n_units must be positive")

    structure_map = ontology.structures.select(
        pl.col("acronym").alias("structure_acronym"),
        "structure_id",
        "structure_name",
        "major_division_id",
        "major_division_acronym",
        "major_division_name",
        "major_division_mapping_status",
    )
    mapped = coverage.select("subject_id", "_nwb_path", "structure_acronym", "n_units").join(
        structure_map, on="structure_acronym", how="left", validate="m:1"
    )
    unknown = mapped.filter(
        (pl.col("structure_acronym") != MISSING_STRUCTURE_ACRONYM)
        & pl.col("structure_id").is_null()
    )
    if unknown.height:
        values = unknown.get_column("structure_acronym").unique().sort().to_list()
        raise RuntimeError(
            f"raw structure acronyms are absent from the pinned author tree: {values}"
        )

    session_presence = (
        mapped.filter(pl.col("major_division_id").is_not_null())
        .select(
            "subject_id",
            "_nwb_path",
            "major_division_id",
            "major_division_acronym",
            "major_division_name",
        )
        .unique()
        .with_columns(
            pl.lit(True).alias("has_any_raw_located_unit_in_division"),
            pl.lit("all_units_presence_only").alias("unit_population"),
            pl.lit("unit_count_magnitude_not_used").alias("count_policy"),
        )
        .sort("subject_id", "_nwb_path", "major_division_id")
    )
    acronym_diagnostics = (
        mapped.with_columns(
            pl.when(pl.col("structure_acronym") == MISSING_STRUCTURE_ACRONYM)
            .then(pl.lit("missing_structure_acronym"))
            .when(pl.col("major_division_id").is_not_null())
            .then(pl.lit("mapped_major_division"))
            .otherwise(pl.lit("outside_official_12_major_divisions"))
            .alias("mapping_status")
        )
        .group_by(
            "structure_acronym",
            "structure_id",
            "structure_name",
            "major_division_id",
            "major_division_acronym",
            "major_division_name",
            "mapping_status",
        )
        .agg(
            pl.col("subject_id").n_unique().alias("n_mice_observed"),
            pl.col("_nwb_path").n_unique().alias("n_sessions_observed"),
        )
        .with_columns(
            pl.lit("all_units_presence_only").alias("unit_population"),
            pl.lit("unit_count_magnitude_not_used").alias("count_policy"),
        )
        .sort("structure_acronym")
    )
    candidate_diagnostics = _build_candidate_diagnostics(mapped, ontology)
    return SessionDivisionMapping(
        session_presence=session_presence,
        acronym_diagnostics=acronym_diagnostics,
        candidate_diagnostics=candidate_diagnostics,
    )


def _cache_pinned_resource(
    resource: PinnedResource,
    destination: pathlib.Path,
    *,
    downloader: Callable[[str], bytes],
) -> str:
    if destination.exists():
        _validate_payload(destination.read_bytes(), resource)
        destination.chmod(0o444)
        return "verified_existing"
    payload = downloader(resource.url)
    _validate_payload(payload, resource)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = pathlib.Path(temporary_name)
    try:
        temporary.write_bytes(payload)
        temporary.chmod(0o444)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "downloaded_and_verified"


def _validate_payload(payload: bytes, resource: PinnedResource) -> None:
    if len(payload) != resource.size_bytes:
        raise RuntimeError(
            f"{resource.key} size differs: expected={resource.size_bytes}, observed={len(payload)}"
        )
    observed = hashlib.sha256(payload).hexdigest()
    if observed != resource.sha256:
        raise RuntimeError(
            f"{resource.key} SHA-256 differs: expected={resource.sha256}, observed={observed}"
        )


def _download_bytes(url: str, *, timeout_seconds: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-dg-paper/ontology-freeze"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.read()
    except (OSError, urllib.error.URLError) as error:
        raise RuntimeError(f"could not download pinned ontology source {url}: {error}") from error


def _read_author_tree(path: pathlib.Path) -> pl.DataFrame:
    raw = pl.read_csv(path, infer_schema_length=None)
    required = {"id", "acronym", "name", "parent_structure_id", "structure_id_path"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise RuntimeError(f"author CCF tree lacks columns: {missing}")
    structures = (
        raw.filter(pl.col("id").is_not_null())
        .select(
            pl.col("id").cast(pl.Int64).alias("structure_id"),
            pl.col("acronym").cast(pl.String),
            pl.col("name").cast(pl.String).alias("structure_name"),
            pl.col("parent_structure_id").cast(pl.Int64),
            pl.col("structure_id_path").cast(pl.String),
        )
        .sort("structure_id")
    )
    if structures.height != EXPECTED_STRUCTURE_COUNT:
        raise RuntimeError(
            "author CCF tree has "
            f"{structures.height} structures, expected {EXPECTED_STRUCTURE_COUNT}"
        )
    if structures.get_column("structure_id").n_unique() != structures.height:
        raise RuntimeError("author CCF tree structure IDs are not unique")
    if structures.get_column("acronym").n_unique() != structures.height:
        raise RuntimeError("author CCF tree acronyms are not unique")
    return structures


def _read_official_structure_graph(path: pathlib.Path) -> pl.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("success") is not True or not isinstance(payload.get("msg"), list):
        raise RuntimeError("official StructureGraph response is not successful")
    rows: list[dict[str, Any]] = []

    def append_node(node: dict[str, Any], ancestors: tuple[int, ...]) -> None:
        identifier = int(node["id"])
        path_ids = (*ancestors, identifier)
        rows.append(
            {
                "structure_id": identifier,
                "official_acronym": node["acronym"],
                "official_name": node["name"],
                "official_parent_structure_id": node.get("parent_structure_id"),
                "official_structure_id_path": f"/{'/'.join(map(str, path_ids))}/",
            }
        )
        for child in node.get("children", []):
            append_node(child, path_ids)

    for root in payload["msg"]:
        append_node(root, ())
    result = pl.DataFrame(rows, infer_schema_length=None).sort("structure_id")
    if result.height != EXPECTED_STRUCTURE_COUNT:
        raise RuntimeError(
            "official StructureGraph has "
            f"{result.height} nodes, expected {EXPECTED_STRUCTURE_COUNT}"
        )
    if result.get_column("structure_id").n_unique() != result.height:
        raise RuntimeError("official StructureGraph IDs are not unique")
    return result


def _read_official_structure_set(path: pathlib.Path) -> pl.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("success") is not True or payload.get("total_rows") != 12:
        raise RuntimeError("official major-division structure-set response is not 12 rows")
    rows = payload.get("msg")
    if not isinstance(rows, list) or len(rows) != 12:
        raise RuntimeError("official major-division structure set lacks 12 structures")
    result = pl.DataFrame(
        [
            {
                "major_division_order": index,
                "major_division_id": int(row["id"]),
                "major_division_acronym": row["acronym"],
                "major_division_name": row["name"],
                "official_structure_id_path": row["structure_id_path"],
                "structure_graph_id": STRUCTURE_GRAPH_ID,
                "structure_set_id": MAJOR_DIVISION_STRUCTURE_SET_ID,
            }
            for index, row in enumerate(rows, start=1)
        ],
        infer_schema_length=None,
    )
    return result


def _validate_author_against_official(author: pl.DataFrame, official: pl.DataFrame) -> None:
    comparison = author.join(official, on="structure_id", how="inner", validate="1:1")
    if comparison.height != EXPECTED_STRUCTURE_COUNT:
        raise RuntimeError("author and official StructureGraph ID sets differ")
    parent_conflicts = comparison.filter(
        ~pl.col("parent_structure_id").eq_missing(pl.col("official_parent_structure_id"))
    )
    if parent_conflicts.height:
        raise RuntimeError("author and official StructureGraph parent links differ")
    path_conflicts = comparison.filter(
        pl.col("structure_id_path") != pl.col("official_structure_id_path")
    )
    if path_conflicts.height:
        raise RuntimeError("author and official StructureGraph ancestry paths differ")


def _validate_divisions(
    divisions: pl.DataFrame,
    author: pl.DataFrame,
    official: pl.DataFrame,
) -> None:
    expected = {
        identifier: (acronym, name) for identifier, acronym, name in EXPECTED_MAJOR_DIVISIONS
    }
    observed = {
        row["major_division_id"]: (
            row["major_division_acronym"],
            row["major_division_name"],
        )
        for row in divisions.iter_rows(named=True)
    }
    if observed != expected:
        raise RuntimeError(f"official structure-set membership differs: {observed}")
    check = divisions.join(
        author.select(
            pl.col("structure_id").alias("major_division_id"),
            pl.col("acronym").alias("author_acronym"),
            "structure_id_path",
        ),
        on="major_division_id",
        how="inner",
        validate="1:1",
    ).join(
        official.select(
            pl.col("structure_id").alias("major_division_id"),
            "official_acronym",
            pl.col("official_structure_id_path").alias("graph_structure_id_path"),
        ),
        on="major_division_id",
        how="inner",
        validate="1:1",
    )
    if check.height != len(EXPECTED_MAJOR_DIVISIONS):
        raise RuntimeError("one or more official major divisions are absent from the trees")
    conflicts = check.filter(
        (pl.col("major_division_acronym") != pl.col("author_acronym"))
        | (pl.col("major_division_acronym") != pl.col("official_acronym"))
        | (pl.col("official_structure_id_path") != pl.col("structure_id_path"))
        | (pl.col("official_structure_id_path") != pl.col("graph_structure_id_path"))
    )
    if conflicts.height:
        raise RuntimeError("major-division acronym or ancestry differs across sources")


def _validate_expected_candidates(
    structures: pl.DataFrame,
    divisions: pl.DataFrame,
    official: pl.DataFrame,
) -> None:
    expected_ids = {acronym: identifier for identifier, acronym, _ in EXPECTED_MAJOR_DIVISIONS}
    for acronym in EXPECTED_CANDIDATE_ACRONYMS:
        identifier = expected_ids[acronym]
        author_row = structures.filter(pl.col("acronym") == acronym)
        official_row = official.filter(pl.col("official_acronym") == acronym)
        set_row = divisions.filter(pl.col("major_division_acronym") == acronym)
        if author_row.height != 1 or official_row.height != 1 or set_row.height != 1:
            raise RuntimeError(f"expected division candidate {acronym} is not unique")
        row = author_row.row(0, named=True)
        if row["structure_id"] != identifier or row["major_division_id"] != identifier:
            raise RuntimeError(f"expected division candidate {acronym} does not map to itself")


def _build_candidate_diagnostics(
    mapped: pl.DataFrame,
    ontology: MajorDivisionOntology,
) -> pl.DataFrame:
    observed = mapped.group_by("structure_acronym").agg(
        pl.col("subject_id").n_unique().alias("n_mice_observed"),
        pl.col("_nwb_path").n_unique().alias("n_sessions_observed"),
    )
    expected_ids = {acronym: identifier for identifier, acronym, _ in EXPECTED_MAJOR_DIVISIONS}
    rows = []
    for acronym in EXPECTED_CANDIDATE_ACRONYMS:
        structure = ontology.structures.filter(pl.col("acronym") == acronym).row(0, named=True)
        counts = observed.filter(pl.col("structure_acronym") == acronym)
        rows.append(
            {
                "candidate_acronym": acronym,
                "expected_structure_id": expected_ids[acronym],
                "author_structure_id": structure["structure_id"],
                "mapped_major_division_id": structure["major_division_id"],
                "mapped_major_division_acronym": structure["major_division_acronym"],
                "observed_in_m0_coverage": counts.height == 1,
                "n_mice_observed": (counts.item(0, "n_mice_observed") if counts.height else 0),
                "n_sessions_observed": (
                    counts.item(0, "n_sessions_observed") if counts.height else 0
                ),
                "mapping_status": "pass_exact_id_ancestry_and_set_membership",
                "observation_status": (
                    "observed_as_raw_m0_acronym"
                    if counts.height
                    else "not_observed_as_raw_m0_acronym"
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _parse_structure_id_path(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(part) for part in value.strip("/").split("/") if part)
    except (AttributeError, ValueError) as error:
        raise RuntimeError(f"invalid structure_id_path {value!r}") from error
    if not result:
        raise RuntimeError(f"empty structure_id_path {value!r}")
    return result
