import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_huggingface import HuggingFaceEmbeddings
from src.config import EMBEDDING_MODEL_NAME

def get_embedding_model():
  return HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME
  )