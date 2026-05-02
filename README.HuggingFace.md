---
title: Document Retrieval QA
emoji: 📄
colorFrom: indigo
colorTo: violet
sdk: gradio
app_file: app.py
pinned: false
---

# Document Retrieval QA

Ask questions about **your own** `.txt`, `.pdf`, or `.docx` files. The Space **retrieves** the most relevant passages with **BM25**, then runs **extractive question answering** with a **[RoBERTa-base model](https://arxiv.org/pdf/1907.11692)** fine-tuned on **[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)** ([`IProject-10/roberta-base-finetuned-squad2`](https://huggingface.co/IProject-10/roberta-base-finetuned-squad2)).

## How to use this Space

1. **Upload** one or more documents (text-based PDFs work best; scanned PDFs need OCR elsewhere first).
2. **Type your question** in plain language (a question mark is optional).
3. Press **Submit**. You’ll get:
   - an **answer** extracted from the text,
   - the **retrieved passage** used as context,
   - **scores** (model confidence and BM25 for that chunk).

## What happens under the hood

1. Each file’s text is **split into chunks** so retrieval and QA work better on long documents.
2. **BM25** ranks chunks against your query; the **top matches** are sent to the QA model.
3. The model predicts a **span** from those chunks (classic extractive QA, not open-ended chat).

## Tips & limitations

- Answers only appear if something in your uploads reasonably matches the question; **garbage in → weak answers**.
- The model predicts a **substring-style answer**, not guaranteed reasoning across many pages in one shot.
- **Very large uploads** increase CPU time; for heavy use, duplicate this Space with a **GPU** hardware preset.

## Run locally

If you mirror this repo, create a virtual environment and install from `requirements.txt`, then:

```bash
python app.py
```

## Project lineage

College project extending ideas from encoder-based QA and retrieval workflows; conceptually aligned with comparative work on building QA systems from pre-trained encoders.
