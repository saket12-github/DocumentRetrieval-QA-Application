"""Read text from user-uploaded documents (.txt, .docx, .pdf)."""

from __future__ import annotations

import os

import pdfplumber
from docx import Document

from config import ALLOWED_EXTENSIONS


def extract_text_from_file(path: str) -> str:
    """Read plaintext from a supported file path. Raises ValueError on failure."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format {ext or '(none)'}. "
            f"Use: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if ext == ".txt":
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read()

    if ext == ".docx":
        return _extract_docx(path)

    if ext == ".pdf":
        fragments: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                fragments.append(page_text if page_text else "")
        return "\n".join(fragments)

    raise ValueError(f"Unhandled extension: {ext}")


def _extract_docx(path: str) -> str:
    """Paragraphs plus simple table text (rows joined with tabs)."""
    document = Document(path)
    parts: list[str] = []
    for para in document.paragraphs:
        if para.text and para.text.strip():
            parts.append(para.text.strip())
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text and c.text.strip()]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)
