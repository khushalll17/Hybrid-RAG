import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_and_split_packages():

  loader = TextLoader(f"{DATA_DIR}/packages.txt", encoding="utf-8")
  documents = loader.load()

  splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
  )
  return splitter.split_documents(documents)