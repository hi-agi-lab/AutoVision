# Copyright (c) 2026 AutoVision contributors. Apache-2.0.
"""Memory-mapped file-list serialization via PyArrow IPC.

The dataset row list is written once (by local rank zero) to an IPC file and
re-opened memory-mapped by every process: zero-copy sharing across dataloader
workers, no data through pickle (``__getstate__`` carries only the path).
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar, overload

import pyarrow as pa
import pyarrow.ipc as ipc

__all__ = ["MemoryMappedSequence", "write_items_to_file"]

T = TypeVar("T")

_CHUNK_SIZE = 10_000


def _infer_type(value: Any) -> pa.DataType:
    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    return pa.string()


def _schema_from_item(item: dict[str, Any]) -> tuple[pa.Schema, list[str]]:
    column_names = list(item.keys())
    schema = pa.schema([pa.field(name, _infer_type(item[name])) for name in column_names])
    return schema, column_names


def _stream_write_table_to_file(
    items: Iterator[dict[str, Any]],
    mmap_filepath,
    schema: pa.Schema,
    column_names: list[str],
) -> None:
    with ipc.new_file(sink=mmap_filepath, schema=schema) as writer:
        chunk: dict[str, list[Any]] = {name: [] for name in column_names}
        count_in_chunk = 0
        for item in items:
            for name in column_names:
                chunk[name].append(item[name])
            count_in_chunk += 1
            if count_in_chunk >= _CHUNK_SIZE:
                writer.write_table(pa.table(chunk))
                chunk = {name: [] for name in column_names}
                count_in_chunk = 0
        if count_in_chunk > 0:
            writer.write_table(pa.table(chunk))


def write_items_to_file(items: Iterable[dict[str, Any]], mmap_filepath) -> None:
    """Write items to a PyArrow IPC file (streaming, chunked)."""
    iterator = iter(items)
    first_item = next(iterator, None)
    if first_item is None:
        schema = pa.schema([pa.field("empty", pa.string())])
        with ipc.new_file(sink=mmap_filepath, schema=schema):
            pass
        return
    schema, column_names = _schema_from_item(first_item)
    _stream_write_table_to_file(
        _chain_first(first_item, iterator), mmap_filepath, schema, column_names
    )


def _chain_first(first_item: dict[str, Any], rest: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    yield first_item
    yield from rest


def _mmap_table_from_file(mmap_filepath) -> pa.Table:
    try:
        with pa.memory_map(str(mmap_filepath)) as source:
            return ipc.open_file(source).read_all()
    except (pa.ArrowInvalid, OSError) as error:
        if isinstance(error, FileNotFoundError | PermissionError):
            # Distinct operational failures must keep their own identity:
            # calling them "corrupt" would mislead troubleshooting.
            raise
        # A truncated or clobbered IPC file (crashed writer on an old
        # format, manual edit) must fail with a rebuild hint, not a raw
        # Arrow error from deep inside deserialization.
        raise RuntimeError(
            f"Memory-mapped dataset file '{mmap_filepath}' is corrupt or "
            f"truncated ({error}). Rebuild it: delete the file, or re-run "
            f"with overwrite=True / a fresh output directory."
        ) from error


class MemoryMappedSequence(Sequence[dict[str, Any]]):
    """Zero-copy random access to a serialized item sequence.

    The Arrow table is re-opened lazily per process (keyed on ``os.getpid()``)
    so both fork and spawn dataloader workers are safe; pickling carries only
    ``{"path", "columns"}``.
    """

    def __init__(self, path, columns: list[str]) -> None:
        self.path = path
        self.columns = list(columns)
        self._pid: int | None = None
        self._table: pa.Table | None = None

    @classmethod
    def from_file(cls, mmap_filepath) -> "MemoryMappedSequence":
        table = _mmap_table_from_file(mmap_filepath)
        return cls(mmap_filepath, table.column_names)

    def table(self) -> pa.Table:
        if self._table is None or self._pid != os.getpid():
            self._table = _mmap_table_from_file(self.path)
            self._pid = os.getpid()
        return self._table

    def __len__(self) -> int:
        return self.table().num_rows

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index):
        table = self.table().select(self.columns)
        if isinstance(index, int):
            length = table.num_rows
            if index < 0:
                index += length
            if not 0 <= index < length:
                raise IndexError(f"Index {index} out of range for length {length}")
            return table.slice(index, 1).to_pylist()[0]
        if isinstance(index, slice):
            start, stop, step = index.indices(table.num_rows)
            if step == 1:
                return table.slice(start, max(0, stop - start)).to_pylist()
            return table.take(list(range(start, stop, step))).to_pylist()
        raise TypeError(f"Invalid index type: {type(index).__name__}")

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path, "columns": self.columns}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.path = state["path"]
        self.columns = state["columns"]
        self._pid = None
        self._table = None
