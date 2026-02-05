import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.vectorstores import FAISS
from src.embeddings import load_embedding_model
from src.rag_loader import load_and_split_packages
from src.config import VECTOR_DB_DIR

embeddings = load_embedding_model()
documents = load_and_split_packages()

if not documents:
  print("No documents loaded for RAG. packages.txt may be empty.")
    
vectorstore = FAISS.from_documents(documents, embeddings)

os.makedirs(VECTOR_DB_DIR, exist_ok=True)
vectorstore.save_local(VECTOR_DB_DIR)

print(f"Vector store saved at: {VECTOR_DB_DIR}")