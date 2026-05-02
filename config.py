"""Central settings for Document Retrieval QA (Hugging Face Space)."""

MODEL_NAME = "IProject-10/roberta-base-finetuned-squad2"
TOP_K_CHUNKS = 5
MAX_CHUNK_CHARS = 1800
ALLOWED_EXTENSIONS = frozenset({".txt", ".docx", ".pdf"})
