"""
Document Retrieval QA — Hugging Face Space entrypoint.

BM25 retrieves relevant chunks from uploads; extractive QA (RoBERTa) picks an answer span.
See `answerer.py` for orchestration and `qa_model.py` for model loading.
"""

from __future__ import annotations

import logging

from ui import create_demo

logging.basicConfig(level=logging.INFO)

demo = create_demo()

if __name__ == "__main__":
    demo.queue(max_size=16).launch(server_name="0.0.0.0", server_port=7860)
