"""Common Document representation and raw corpus iterators."""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Document:
    """Unified document format for the Siat pretraining data pipeline."""

    text: str
    source: str
    language: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def make_document_id(source: str, text: str) -> str:
    """Deterministic SHA256 document id (not Python ``hash()``)."""
    payload = f"{source}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def content_hash(text: str) -> str:
    """SHA256 of normalized text for exact deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def list_input_files(path: str | Path, suffix: str) -> list[Path]:
    """Return sorted files with ``suffix`` under a file or directory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Corpus path does not exist: {path.resolve()}")
    suffix = suffix.lower()
    if path.is_file():
        if path.suffix.lower() != suffix:
            raise ValueError(
                f"Expected a {suffix} file, got {path.name!r}."
            )
        return [path]
    if path.is_dir():
        files = sorted(
            p for p in path.rglob(f"*{suffix}") if p.is_file()
        )
        if not files:
            raise FileNotFoundError(
                f"No {suffix} files found under: {path.resolve()}"
            )
        return files
    raise ValueError(f"Path is neither file nor directory: {path}")


def iter_txt_documents(
    path: str | Path,
    *,
    source: str,
    language: str | None = None,
    encoding_errors: str = "replace",
) -> Iterator[Document]:
    """Yield one Document per ``.txt`` file (sorted)."""
    for file_path in list_input_files(path, ".txt"):
        raw = file_path.read_text(encoding="utf-8", errors=encoding_errors)
        yield Document(
            text=raw,
            source=source,
            language=language,
            metadata={
                "path": str(file_path.as_posix()),
                "format": "txt",
            },
        )


def iter_jsonl_documents(
    path: str | Path,
    *,
    source: str,
    text_field: str = "text",
    language: str | None = None,
    encoding_errors: str = "replace",
) -> Iterator[tuple[Document | None, str | None]]:
    """Yield ``(Document, None)`` or ``(None, error_reason)`` per JSONL line.

    Malformed lines do not abort the whole file.
    """
    for file_path in list_input_files(path, ".jsonl"):
        with file_path.open("r", encoding="utf-8", errors=encoding_errors) as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except json.JSONDecodeError:
                    yield None, f"malformed_json:{file_path.name}:{line_no}"
                    continue
                if not isinstance(obj, dict):
                    yield None, f"not_object:{file_path.name}:{line_no}"
                    continue
                if text_field not in obj:
                    yield None, f"missing_field:{file_path.name}:{line_no}"
                    continue
                text = obj[text_field]
                if not isinstance(text, str):
                    yield None, f"non_string_text:{file_path.name}:{line_no}"
                    continue
                meta = {
                    "path": str(file_path.as_posix()),
                    "format": "jsonl",
                    "line": line_no,
                }
                for key in ("license", "url", "notes", "id"):
                    if key in obj:
                        meta[key] = obj[key]
                yield (
                    Document(
                        text=text,
                        source=source,
                        language=language,
                        metadata=meta,
                    ),
                    None,
                )


def iter_parquet_documents(
    path: str | Path,
    *,
    source: str,
    text_field: str = "text",
    language: str | None = None,
    batch_size: int = 1024,
    provenance_fields: tuple[str, ...] = (
        "id",
        "url",
        "dump",
        "title",
        "language",
        "language_score",
        "token_count",
        "score",
        "int_score",
    ),
) -> Iterator[tuple[Document | None, str | None]]:
    """Stream Documents from one or more ``.parquet`` files via pyarrow batches.

    Does **not** load the full table into pandas/memory. Schema is inspected
    first; ``text_field`` must exist or a ``ValueError`` is raised.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "Reading parquet requires pyarrow. Install with: pip install pyarrow"
        ) from e

    files = list_input_files(path, ".parquet")
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        names = list(pf.schema_arrow.names)
        if text_field not in names:
            raise ValueError(
                f"Parquet {file_path.name} has no column {text_field!r}. "
                f"Available columns: {names}"
            )
        keep_meta = [c for c in provenance_fields if c in names]
        columns = [text_field] + keep_meta
        for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
            texts = batch.column(text_field)
            n = batch.num_rows
            meta_cols = {
                c: batch.column(c) for c in keep_meta
            }
            for i in range(n):
                val = texts[i]
                if val is None or val.as_py() is None:
                    yield None, f"null_text:{file_path.name}"
                    continue
                text = val.as_py()
                if not isinstance(text, str):
                    yield None, f"non_string_text:{file_path.name}"
                    continue
                meta: dict[str, Any] = {
                    "path": str(file_path.as_posix()),
                    "format": "parquet",
                }
                for c, col in meta_cols.items():
                    cell = col[i]
                    if cell is not None and cell.as_py() is not None:
                        meta[c] = cell.as_py()
                yield (
                    Document(
                        text=text,
                        source=source,
                        language=language,
                        metadata=meta,
                    ),
                    None,
                )


def iter_source_documents(
    path: str | Path,
    *,
    source: str,
    fmt: str,
    text_field: str = "text",
    language: str | None = None,
) -> Iterator[tuple[Document | None, str | None]]:
    """Dispatch by format; always yields ``(doc|None, error|None)``."""
    fmt = fmt.lower()
    if fmt == "txt":
        for doc in iter_txt_documents(
            path, source=source, language=language
        ):
            yield doc, None
        return
    if fmt == "jsonl":
        yield from iter_jsonl_documents(
            path,
            source=source,
            text_field=text_field,
            language=language,
        )
        return
    if fmt == "parquet":
        yield from iter_parquet_documents(
            path,
            source=source,
            text_field=text_field,
            language=language,
        )
        return
    raise ValueError(
        f"Unsupported corpus format: {fmt!r} (use txt, jsonl, or parquet)."
    )

def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load a JSON manifest with a ``sources`` list."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "sources" not in data:
        raise ValueError("Manifest must be an object with a 'sources' array.")
    if not isinstance(data["sources"], list) or not data["sources"]:
        raise ValueError("Manifest 'sources' must be a non-empty list.")
    for i, src in enumerate(data["sources"]):
        for key in ("name", "path", "format", "weight"):
            if key not in src:
                raise ValueError(f"sources[{i}] missing required field {key!r}.")
        if float(src["weight"]) < 0:
            raise ValueError(f"sources[{i}].weight must be >= 0.")
    return data
