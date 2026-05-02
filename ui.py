"""Gradio layout for Hugging Face Spaces."""

from __future__ import annotations

import gradio as gr

from answerer import answer_question

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


def create_demo():
    theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#eef2ff",
            c100="#e0e7ff",
            c200="#c7d2fe",
            c300="#a5b4fc",
            c400="#818cf8",
            c500="#6366f1",
            c600="#4f46e5",
            c700="#4338ca",
            c800="#3730a3",
            c900="#312e81",
            c950="#1e1b4b",
        ),
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    )

    return gr.Interface(
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
