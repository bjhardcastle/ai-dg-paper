"""Obstore-only accessors for NWB columns not yet materialized by ``scan_nwb``.

The pinned lazynwb custom reader directly materializes numeric scalar columns
and numeric ragged columns such as ``spike_times``.  Its metadata parser also
understands HDF5 variable-length string references, but dev8 does not yet wire
that type into the public table materializer.  This module provides that small
bridge so analyses can enforce string-valued unit quality labels without
opening a legacy file accessor.

The implementation intentionally uses the custom reader bundled with the
exact lazynwb revision pinned by this repository.  It reads only byte ranges
through obstore and delegates HDF5 metadata and global-heap decoding to
lazynwb's parser.
"""

from __future__ import annotations

import asyncio
import collections.abc
import concurrent.futures
import os
from typing import Any

import lazynwb
import lazynwb._hdf5.parser
import lazynwb._hdf5.reader
import lazynwb.tables
import polars as pl

PathLike = str | os.PathLike[str]


def map_indexed_numeric_column_batches(
    source: PathLike,
    table_path: str,
    column_name: str,
    row_indices: collections.abc.Sequence[int],
    callback: collections.abc.Callable[[tuple[int, ...], list[list[Any]]], None],
    *,
    batch_size: int = 64,
) -> None:
    """Read a ragged numeric column in bounded-memory batches.

    One lazynwb custom-reader lifecycle is reused for every batch. ``callback``
    is invoked immediately with the table indices and decoded rows, allowing a
    caller to reduce each batch before the next byte ranges are requested.
    """

    exact_table_path = table_path.strip("/")
    if not exact_table_path:
        raise ValueError("table_path must identify an exact NWB table")
    if not column_name or "/" in column_name:
        raise ValueError("column_name must be one dataset name")
    selected_rows = tuple(row_indices)
    _validate_row_indices(selected_rows)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if not callable(callback):
        raise TypeError("callback must be callable")

    lazynwb.config.anon = True
    lazynwb.config.use_obstore = True
    lazynwb.config.use_remfile = False
    _run_async(
        _map_indexed_numeric_column_batches(
            source=str(source),
            exact_table_path=exact_table_path,
            column_name=column_name,
            row_indices=selected_rows,
            callback=callback,
            batch_size=batch_size,
        )
    )


def read_vlen_string_column(
    source: PathLike,
    table_path: str,
    column_name: str,
    *,
    row_indices: collections.abc.Sequence[int] | None = None,
) -> pl.DataFrame:
    """Read one contiguous variable-length string column through obstore.

    Parameters
    ----------
    source
        A local or cloud NWB object understood by lazynwb's obstore reader.
    table_path
        Exact NWB table path, with or without a leading slash.
    column_name
        Variable-length string dataset within the table.
    row_indices
        Optional zero-based table rows. Returned rows preserve this order.

    Returns
    -------
    polars.DataFrame
        ``_table_index``, the requested string column, and ``_nwb_path``.
    """

    exact_table_path = table_path.strip("/")
    if not exact_table_path:
        raise ValueError("table_path must identify an exact NWB table")
    if not column_name or "/" in column_name:
        raise ValueError("column_name must be one dataset name")

    selected_rows = None if row_indices is None else tuple(row_indices)
    if selected_rows is not None:
        _validate_row_indices(selected_rows)

    lazynwb.config.anon = True
    lazynwb.config.use_obstore = True
    lazynwb.config.use_remfile = False
    values, normalized_rows = _run_async(
        _read_vlen_string_column(
            source=str(source),
            exact_table_path=exact_table_path,
            column_name=column_name,
            row_indices=selected_rows,
        )
    )
    return pl.DataFrame(
        {
            "_table_index": normalized_rows,
            column_name: values,
            "_nwb_path": [str(source)] * len(values),
        },
        schema={
            "_table_index": pl.UInt32,
            column_name: pl.String,
            "_nwb_path": pl.String,
        },
    )


async def _map_indexed_numeric_column_batches(
    *,
    source: str,
    exact_table_path: str,
    column_name: str,
    row_indices: tuple[int, ...],
    callback: collections.abc.Callable[[tuple[int, ...], list[list[Any]]], None],
    batch_size: int,
) -> None:
    reader = lazynwb._hdf5.reader._default_hdf5_backend_reader(source)
    try:
        snapshot = await reader.read_table_schema_snapshot(exact_table_path)
        columns = {column.name: column for column in snapshot.columns}
        data_column = columns.get(column_name)
        if data_column is None:
            raise KeyError(f"{column_name!r} is absent from {exact_table_path!r}")
        index_name = data_column.index_column_name or f"{column_name}_index"
        index_column = columns.get(index_name)
        if index_column is None:
            raise KeyError(f"index column {index_name!r} is absent from {exact_table_path!r}")
        plan = lazynwb.tables._plan_direct_hdf5_table_reads((data_column, index_column))
        if (
            len(plan.indexed_columns) != 1
            or plan.indexed_columns[0].data_column.name != column_name
            or plan.fallback_columns
        ):
            raise NotImplementedError(
                f"{exact_table_path}/{column_name} is not a directly readable "
                "numeric indexed column"
            )
        row_count = int(data_column.shape[0]) if data_column.shape else 0
        if any(row >= row_count for row in row_indices):
            raise IndexError(f"row index exceeds {row_count - 1}")
        indexed_plan = plan.indexed_columns[0]
        for start in range(0, len(row_indices), batch_size):
            batch_rows = row_indices[start : start + batch_size]
            values = await lazynwb.tables._read_direct_hdf5_indexed_column(
                reader._range_reader,
                indexed_plan,
                batch_rows,
            )
            callback(batch_rows, values)
            del values
    finally:
        await reader.close()


async def _read_vlen_string_column(
    *,
    source: str,
    exact_table_path: str,
    column_name: str,
    row_indices: tuple[int, ...] | None,
) -> tuple[list[str], list[int]]:
    reader = lazynwb._hdf5.reader._default_hdf5_backend_reader(source)
    try:
        identity = await reader.get_source_identity()
        if identity.content_length is None:
            raise RuntimeError("obstore source did not report a content length")
        snapshot = await reader.read_table_schema_snapshot(exact_table_path)
        matches = [column for column in snapshot.columns if column.name == column_name]
        if len(matches) != 1:
            raise KeyError(
                f"expected one {column_name!r} column at {exact_table_path!r}; found {len(matches)}"
            )
        column = matches[0]
        dataset = column.dataset
        if column.dtype.kind != "vlen_string":
            raise TypeError(
                f"{exact_table_path}/{column_name} is {column.dtype.kind!r}, "
                "not a variable-length string column"
            )
        if "direct_contiguous" not in dataset.read_capabilities:
            raise NotImplementedError(
                f"{exact_table_path}/{column_name} is not stored contiguously"
            )
        if dataset.hdf5_data_offset is None or dataset.hdf5_storage_size is None:
            raise RuntimeError(f"{exact_table_path}/{column_name} lacks direct byte-range metadata")
        if not column.shape or column.ndim != 1:
            raise NotImplementedError(
                f"{exact_table_path}/{column_name} must be a one-dimensional dataset"
            )

        row_count = int(column.shape[0])
        normalized_rows = list(range(row_count)) if row_indices is None else list(row_indices)
        if len(set(normalized_rows)) != len(normalized_rows):
            raise ValueError("row_indices must not contain duplicates")
        if any(row >= row_count for row in normalized_rows):
            raise IndexError(f"row index exceeds {row_count - 1}")

        scanner = reader._get_scanner(int(identity.content_length))
        superblock = await scanner.bootstrap()
        payload = await reader._range_reader.read_range(
            dataset.hdf5_data_offset,
            length=dataset.hdf5_storage_size,
        )
        references = lazynwb._hdf5.parser._parse_vlen_string_references(
            payload,
            count=row_count,
            offset_size=superblock.offset_size,
        )
        values = [
            await scanner._read_global_heap_string(references[row_index])
            for row_index in normalized_rows
        ]
        return values, normalized_rows
    finally:
        await reader.close()


def _validate_row_indices(row_indices: tuple[int, ...]) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) for value in row_indices):
        raise TypeError("row_indices must contain integers")
    if any(value < 0 for value in row_indices):
        raise ValueError("row_indices must be non-negative")
    if len(set(row_indices)) != len(row_indices):
        raise ValueError("row_indices must not contain duplicates")


def _run_async(coroutine: Any) -> Any:
    """Run a coroutine from ordinary code or a thread with an active loop."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coroutine).result()
