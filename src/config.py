import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3