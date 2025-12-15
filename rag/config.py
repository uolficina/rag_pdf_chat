import os

MISTRAL_MODEL = "mistral-small-latest"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
CROSSENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

BASE_DIR = "index"
os.makedirs(BASE_DIR, exist_ok=True)

# Shared state to avoid scattered globals
state = {
    "embed_model": None,
    "index": None,
    "cross_encoder": None,
    "chunks": [],
    "chunk_texts": [],
    "total_pages": 0,
    "page_texts": [],
}

