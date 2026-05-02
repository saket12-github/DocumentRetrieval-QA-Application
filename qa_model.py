"""Load Hugging Face extractive QA model and run inference on one passage."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from config import MODEL_NAME

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
qa_model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
qa_model.to(device)
qa_model.eval()

_pipeline_device = device.index if device.type == "cuda" else -1
_retrieval_qa_pipeline = pipeline(
    "question-answering",
    model=qa_model,
    tokenizer=tokenizer,
    device=_pipeline_device,
)


def run_qa_on_passage(question: str, passage: str) -> dict[str, Any]:
    """
    Run extractive QA on a single passage.

    Uses extra kwargs when the installed transformers supports them (e.g. SQuAD 2.0 handling).
    """
    base = {"question": question, "context": passage, "truncation": True}
    try:
        out = _retrieval_qa_pipeline(
            **base,
            max_answer_len=64,
            top_k=1,
            handle_impossible_answer=True,
        )
    except TypeError:
        out = _retrieval_qa_pipeline(**base)

    payload = out[0] if isinstance(out, list) else out
    return payload if isinstance(payload, dict) else {}
