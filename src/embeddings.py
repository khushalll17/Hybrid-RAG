from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

def load_embedding_model():
  return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")