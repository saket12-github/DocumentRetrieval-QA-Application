"""BM25 retrieval + extractive QA orchestration."""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from chunking import load_chunks_from_files, normalize_file_list
from config import TOP_K_CHUNKS
from qa_model import run_qa_on_passage

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def highlight_answer(context: str, answer: str) -> str:
    """Mark the extracted span in context when an exact substring match exists."""
    if not answer.strip():
        return context

    idx = context.find(answer)
    if idx != -1:
        end = idx + len(answer)
        return f"{context[:idx]}━━━━━━━━ «{context[idx:end]}» ━━━━━━━━{context[end:]}"

    norm_ctx = _WHITESPACE.sub(" ", context)
    norm_ans = _WHITESPACE.sub(" ", answer.strip())
    if norm_ans and norm_ans in norm_ctx:
        return (
            context
            + "\n\n*(Answer spans may differ slightly from source whitespace; "
            "see Answer field.)*"
        )

    return context + "\n\n*(Could not align answer span in context text.)*"


def answer_question(question: str, files: list[Any] | None) -> tuple[str, str, str]:
    """Run BM25 retrieval + extractive QA. Returns answer, highlighted context, score summary."""
    files = normalize_file_list(files)
    warn_prefix = ""

    try:
        q = (question or "").strip()
        if not q:
            raise ValueError("Please enter a question.")

        passages, warns = load_chunks_from_files(files)
        if warns:
            warn_prefix = "**Note:** " + " ".join(warns) + "\n\n"

        bm25 = BM25Okapi([p.split() for p in passages])
        tokenized_query = q.split()
        bm25_scores_list = bm25.get_scores(tokenized_query)
        n = min(TOP_K_CHUNKS, len(passages))
        ranked_indices = sorted(
            range(len(passages)),
            key=lambda i: bm25_scores_list[i],
            reverse=True,
        )[:n]

        best: dict[str, Any] | None = None

        for passage_idx in ranked_indices:
            passage = passages[passage_idx]
            qa_out = run_qa_on_passage(q, passage)
            qa_score = float(qa_out.get("score", 0.0))
            bm25_score = float(bm25_scores_list[passage_idx])

            row: dict[str, Any] = {
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

        answer_text = (
            best["answer"]
            or "(No span found in top passages; try rephrasing or uploading more relevant text.)"
        )
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
