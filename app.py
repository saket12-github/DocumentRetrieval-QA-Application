import os
from rank_bm25 import BM25Okapi
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch
import gradio as gr
from docx import Document
import pdfplumber

# Load the fine-tuned BERT-based QA model and tokenizer
model_name = "IProject-10/roberta-base-finetuned-squad2"  # Replace with your model name
qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set up the device for BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
qa_model.to(device)
qa_model.eval()

# Create a pipeline for retrieval-augmented QA
retrieval_qa_pipeline = pipeline(
    "question-answering",
    model=qa_model,
    tokenizer=tokenizer,
    device=device.index if torch.cuda.is_available() else -1
)

def extract_text_from_file(file):
    # Determine the file extension
    file_extension = os.path.splitext(file.name)[1].lower()
    text = ""

    try:
        if file_extension == ".txt":
            with open(file.name, "r") as f:
                text = f.read()
        elif file_extension == ".docx":
            doc = Document(file.name)
            for para in doc.paragraphs:
                text += para.text + "\n"
        elif file_extension == ".pdf":
            with pdfplumber.open(file.name) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        else:
            raise ValueError("Unsupported file format: {}".format(file_extension))
    except Exception as e:
        text = str(e)
    return text

def load_passages(files):
    passages = []
    for file in files:
        passage = extract_text_from_file(file)
        passages.append(passage)
    return passages

def highlight_answer(context, answer):
    start_index = context.find(answer)
    if start_index != -1:
        end_index = start_index + len(answer)
        highlighted_context = f"{context[:start_index]}_________<<{context[start_index:end_index]}>>_________{context[end_index:]}"
        return highlighted_context
    else:
        return context

def answer_question(question, files):
    try:
        # Load passages from the uploaded files
        passages = load_passages(files)

        # Create an index using BM25
        bm25 = BM25Okapi([passage.split() for passage in passages])

        # Retrieve relevant passages using BM25
        tokenized_query = question.split()
        candidate_passages = bm25.get_top_n(tokenized_query, passages, n=3)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Extract answer using the pipeline for each candidate passage
        answers_with_context = []
        for passage in candidate_passages:
            answer = retrieval_qa_pipeline(question=question, context=passage)
            bm25_score = bm25_scores[passages.index(passage)]
            answer_with_context = {
                "context": passage,
                "answer": answer["answer"],
                "BM25-score": bm25_score  # BM25 confidence score for this passage
            }
            answers_with_context.append(answer_with_context)

        # Choose the answer with the highest model confidence score
        best_answer = max(answers_with_context, key=lambda x: x["BM25-score"])

        # Highlight the answer in the context
        highlighted_context = highlight_answer(best_answer["context"], best_answer["answer"])

        return best_answer["answer"], highlighted_context, best_answer["BM25-score"]
    except Exception as e:
        return str(e), "", ""

# Description
md = """
### Brief Overview of the project:

A Document-Retrieval QA application built by training **[RoBERTa model](https://arxiv.org/pdf/1907.11692)** on **[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/)** dataset for efficient answer extraction and
the system is augmented by using NLP based **[BM25](https://www.researchgate.net/publication/220613776_The_Probabilistic_Relevance_Framework_BM25_and_Beyond)** retriever for information retrieval from a large text corpus.
The project is a brief enhancement and augmentation to the work done in the research paper **Encoder-based LLMs: Building QA systems and Comparative Analysis**.
In this paper we study about BERT and its advanced variants and learn to build an efficient answer extraction QA system from scratch.
The built system can be used in information retrieval system and search engines.

**Objectives of the projects:**
1. Build a simple Answer Extraction QA system using **RoBERTa-base**: The project is deployed public url objective1.
2. Building a Information Retrieval system for data augmentation using **BM25**
3. **Document Retrieval QA** system by merging Answer Extraction QA system and Information retrieval system

### Demonstrating working of the Application:

<div style="text-align: center;">
    <img src="https://i.imgur.com/oYg8y7N.jpeg" alt="Description Image" style="border: 2px solid #000; border-radius: 5px; width: 600px; height: auto; display: block; margin: 0 auto;">
</div>

**Key Features:**
- Fine-tuned **RoBERTa**- Performs **Answer Extraction** from the retrieved document
- **BM25** Retriever- Performs **Information Retrieval** from the text corpus
- Provides answers with **highlighted context**.
- Application displays accurate **answer**, most relevant document **context** and the corresponding **BM25 score** of the passage to the user

**How to Use:**
1. Upload your corpus document(s).
2. Enter your question in the text box followed by a question mark(?).
3. Get the answer with context and corresponding BM25 scores.
"""

# Define Gradio interface
iface = gr.Interface(
    fn=answer_question,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter your question here...", label="Question"),
        gr.Files(label="Upload text, Word, or PDF files")
    ],
    outputs=[
        gr.Textbox(label="Answer"),
        gr.Textbox(label="Context"),
        gr.Textbox(label="BM25 Score")
    ],
    title="Document Retrieval Question Answering Application",
    description=md,
    css="""
    .container { max-width: 800px; margin: auto; }
    .interface-title { font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; }
    .interface-description { font-family: Arial, sans-serif; font-size: 16px; margin-bottom: 20px; }
    .input-textbox, .output-textbox { font-family: Arial, sans-serif; font-size: 14px; }
    .error { color: red; font-family: Arial, sans-serif; font-size: 14px; }
    """
)

# Launch the interface
iface.launch()
