"""
Document Retrieval QA — Gradio app for Hugging Face Spaces.

BM25 retrieves the most relevant text chunks from uploaded documents; a fine-tuned
RoBERTa extractive QA model answers using those chunks as context.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import gradio as gr
import pdfplumber
import torch
from docx import Document
from rank_bm25 import BM25Okapi
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MODEL_NAME = "IProject-10/roberta-base-finetuned-squad2"
TOP_K_CHUNKS = 5
MAX_CHUNK_CHARS = 1800  # Roughly bounded context for RoBERTa-style models on CPU/GPU Spaces
ALLOWED_EXTENSIONS = {".txt", ".docx", ".pdf"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Model load (startup)
# -----------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_model.to(device)
qa_model.eval()

_pipeline_device = device.index if device.type == "cuda" else -1
retrieval_qa_pipeline = pipeline(
    "question-answering",
    model=qa_model,
    tokenizer=tokenizer,
    device=_pipeline_device,
)


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
        document = Document(path)
        parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        return "\n".join(parts)

    if ext == ".pdf":
        fragments: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                fragments.append(page_text if page_text else "")
        return "\n".join(fragments)

    raise ValueError(f"Unhandled extension: {ext}")


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


def load_chunks_from_files(files: list[Any]) -> tuple[list[str], list[str]]:
    """
    Returns (chunks, warnings). Each file expands to one or more chunks.
    Warnings aggregate non-fatal issues (empty file text, unknown names).
    """
    if not files:
        raise ValueError("Please upload at least one document.")

    chunks: list[str] = []
    warnings: list[str] = []

    file_list = list(files)
    for file_obj in file_list:
        path = getattr(file_obj, "name", None)
        raw_name = getattr(file_obj, "orig_name", None) or getattr(file_obj, "name", "upload")
        if not path:
            warnings.append(f"Skipped an upload with no readable path ({raw_name}).")
            continue

        try:
            text = extract_text_from_file(path).strip()
        except (OSError, ValueError, Exception) as exc:  # pdfplumber / docx can raise varied errors
            logger.exception("Failed to read %s", path)
            raise ValueError(f"Could not read “{raw_name}”: {exc}") from exc

        if not text:
            warnings.append(f"No extractable text in “{raw_name}”.")
            continue

        file_chunks = split_into_chunks(text)
        if not file_chunks:
            warnings.append(f"No chunks produced from “{raw_name}” after processing.")
            continue
        chunks.extend(file_chunks)

    if not chunks:
        raise ValueError(
            "No text could be extracted from your uploads. "
            "Try a different PDF (text-based), .txt, or .docx file."
        )

    return chunks, warnings


_WHITESPACE = re.compile(r"\s+")


def run_qa_on_passage(question: str, passage: str) -> dict[str, Any]:
    """
    Invoke the QA pipeline with kwargs that exist across common transformers versions.

    Older stacks may not support Squad2-only flags like handle_impossible_answer.
    """
    base = {"question": question, "context": passage, "truncation": True}
    try:
        out = retrieval_qa_pipeline(
            **base,
            max_answer_len=64,
            top_k=1,
            handle_impossible_answer=True,
        )
    except TypeError:
        out = retrieval_qa_pipeline(**base)
    return out[0] if isinstance(out, list) else out


def highlight_answer(context: str, answer: str) -> str:
    """Mark the extracted span in context when an exact substring match exists."""
    if not answer.strip():
        return context

    idx = context.find(answer)
    if idx != -1:
        end = idx + len(answer)
        return (
            f"{context[:idx]}━━━━━━━━ «{context[idx:end]}» ━━━━━━━━{context[end:]}"
        )

    norm_ctx = _WHITESPACE.sub(" ", context)
    norm_ans = _WHITESPACE.sub(" ", answer.strip())
    if norm_ans and norm_ans in norm_ctx:
        return (
            context
            + "\n\n*(Answer spans may differ slightly from source whitespace; "
            "see Answer field.)*"
        )

    return context + "\n\n*(Could not align answer span in context text.)*"


def answer_question(question: str, files: list[Any] | None):
    """Run BM25 retrieval + extractive QA. Returns answer, highlighted context, score summary."""
    if files is None:
        files = []

    warn_prefix = ""

    try:
        q = (question or "").strip()
        if not q:
            raise ValueError("Please enter a question.")

        if files is not None and not isinstance(files, list):
            files = [files]

        passages, warns = load_chunks_from_files(files)
        if warns:
            warn_prefix = "**Note:** " + " ".join(warns) + "\n\n"

        bm25 = BM25Okapi([p.split() for p in passages])
        tokenized_query = q.split()
        candidate_passages = bm25.get_top_n(tokenized_query, passages, n=min(TOP_K_CHUNKS, len(passages)))

        bm25_scores_map: dict[int, float] = {
            idx: float(score)
            for idx, score in enumerate(bm25.get_scores(tokenized_query))
        }

        best: dict[str, Any] | None = None

        for passage in candidate_passages:
            qa_out = run_qa_on_passage(q, passage)

            qa_score = float(qa_out.get("score", 0.0))
            passage_idx = passages.index(passage)
            bm25_score = bm25_scores_map.get(passage_idx, 0.0)

            row = {
                "context": passage,
                "answer": (qa_out.get("answer") or "").strip(),
                "qa_score": qa_score,
                "bm25_score": bm25_score,
            }

            if best is None:
                best = row
                continue

            if (row["qa_score"], row["bm25_score"]) > (best["qa_score"], best["bm25_score"]):
                best = row

        assert best is not None

        answer_text = best["answer"] or "(No span found in top passages; try rephrasing or uploading more relevant text.)"
        highlighted = highlight_answer(best["context"], best["answer"])
        scores_md = (
            f"QA confidence: **{best['qa_score']:.4f}** · "
            f"BM25 (chunk): **{best['bm25_score']:.4f}**"
        )

        return warn_prefix + answer_text, highlighted, scores_md

    except ValueError as err:
        return str(err), "", ""
    except Exception as exc:
        logger.exception("Unexpected error during QA")
        return f"Something went wrong: {exc}", "", ""


# -----------------------------------------------------------------------------
# Gradio UI (Hugging Face Spaces)
# -----------------------------------------------------------------------------
DESCRIPTION_MD = """
### Document Retrieval QA

This demo combines **BM25 retrieval** over your uploaded documents with **extractive QA**
using a **[RoBERTa-base](https://arxiv.org/pdf/1907.11692) model fine-tuned on [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)**
([model card](https://huggingface.co/IProject-10/roberta-base-finetuned-squad2)).

**How it works**
1. Your files are split into chunks so long PDFs/DOCX files work better with the retriever.
2. BM25 picks the **top passages** matching your question.
3. The QA model selects an **answer span** from those passages.

**Tips**
- Use clear, specific questions (a question mark is optional).
- For best results, upload documents that actually contain the answer.

**Credits**
Derived from coursework on encoder-based QA and retrieval; related to the theme of the paper *Encoder-based LLMs: Building QA systems and Comparative Analysis*.
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(c50="#eef2ff", c100="#e0e7ff", c200="#c7d2fe", c300="#a5b4fc", c400="#818cf8", c500="#6366f1", c600="#4f46e5", c700="#4338ca", c800="#3730a3", c900="#312e81", c950="#1e1b4b"),
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

demo = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(
            lines=2,
            placeholder='e.g. "What year was the merger announced?"',
            label="Question",
        ),
        gr.Files(
            label="Documents",
            file_count="multiple",
            file_types=[".txt", ".pdf", ".docx"],
        ),
    ],
    outputs=[
        gr.Markdown(label="Answer"),
        gr.Textbox(label="Retrieved passage (context)", lines=14, max_lines=20),
        gr.Markdown(label="Scores"),
    ],
    title="Document Retrieval QA",
    description=DESCRIPTION_MD,
    theme=theme,
    css="""
        .contain { max-width: 920px !important; margin: auto !important; }
        footer {visibility: hidden}
    """,
)

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860)
