import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from src.config import LLM_MODEL_NAME
from dotenv import load_dotenv

load_dotenv()

def get_llm():
  model = HuggingFaceEndpoint(
    repo_id=LLM_MODEL_NAME,
    task="text-generation"
  )
  return ChatHuggingFace(llm=model)