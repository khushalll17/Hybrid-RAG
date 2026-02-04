import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model
from src.rag_loader import load_and_split_packages
from src.config import VECTOR_DB_DIR

def get_vectorstore():

  embeddings = get_embedding_model()

  if os.path.exists(VECTOR_DB_DIR):
    return FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

  documents = load_and_split_packages()

  if not documents:
    raise ValueError("No documents loaded for RAG. packages.txt may be empty.")
    
  vectorstore = FAISS.from_documents(documents, embeddings)

  os.makedirs(VECTOR_DB_DIR, exist_ok=True)
  vectorstore.save_local(VECTOR_DB_DIR)

  return vectorstore