"""Split documents into chunks and load chunks from Gradio file uploads."""

from __future__ import annotations

import logging
import re
from typing import Any

from config import MAX_CHUNK_CHARS
from document_io import extract_text_from_file

logger = logging.getLogger(__name__)


def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split document text into overlapping chunks for BM25 + QA."""
    cleaned = text.strip()
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", cleaned) if p.strip()]
    if not paragraphs:
        paragraphs = [cleaned]

    chunks: list[str] = []
    overlap = max(80, max_chars // 10)

    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        start = 0
        while start < len(para):
            end = min(start + max_chars, len(para))
            chunks.append(para[start:end])
            if end >= len(para):
                break
            start = max(end - overlap, start + 1)

    seen: set[str] = set()
    unique: list[str] = []
    for ch in chunks:
        key = ch.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(ch)
    return unique


def normalize_file_list(files: Any) -> list[Any]:
    """Gradio may pass a single file object or a list."""
    if files is None:
        return []
    return list(files) if isinstance(files, list) else [files]


def load_chunks_from_files(files: list[Any]) -> tuple[list[str], list[str]]:
    """
    Expand uploads into retrieval chunks.

    Returns (chunks, warnings). Raises ValueError for missing/empty corpus.
    """
    if not files:
        raise ValueError("Please upload at least one document.")

    chunks: list[str] = []
    warnings: list[str] = []

    for file_obj in files:
        path = getattr(file_obj, "name", None)
        display_name = getattr(file_obj, "orig_name", None) or getattr(file_obj, "name", "upload")
        if not path:
            warnings.append(f"Skipped an upload with no readable path ({display_name}).")
            continue

        try:
            text = extract_text_from_file(path).strip()
        except (OSError, ValueError) as exc:
            logger.exception("Failed to read %s", path)
            raise ValueError(f'Could not read "{display_name}": {exc}') from exc

        if not text:
            warnings.append(f'No extractable text in "{display_name}".')
            continue

        file_chunks = split_into_chunks(text)
        if not file_chunks:
            warnings.append(f'No chunks produced from "{display_name}" after processing.')
            continue
        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(
            "No text could be extracted from your uploads. "
            "Try a different PDF (text-based), .txt, or .docx file."
        )

    return chunks, warnings
