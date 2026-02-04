import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb", "packages_faiss")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3