import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from src.flight_logic import select_flight
from src.vectorstore import get_vectorstore
from src.llm_setup import get_llm
from src.config import TOP_K, DATA_DIR

def _extract_people(query: str) -> int:
  match = re.search(r"(\d+)\s*(people|persons|travellers|travelers)", query.lower())
  return int(match.group(1)) if match else 1


def _extract_package_tier(query: str) -> str:
  return "Luxury" if "luxury" in query.lower() else "Standard"


def extract_package_price(rag_context: str, tier: str) -> int:
  if tier.lower() == "luxury":
    match = re.search(r"Luxury Tier:\s*\$([0-9]+)", rag_context, re.IGNORECASE)
  else:
    match = re.search(r"Standard Tier:\s*\$([0-9]+)", rag_context, re.IGNORECASE)

  if not match:
    raise ValueError("Package price not found in retrieved brochure text.")

  return int(match.group(1))


def extract_package_price_with_fallback(rag_context: str, tier: str) -> int:
  try:
    return extract_package_price(rag_context, tier)
  except ValueError:
    pass  

  with open(os.path.join(DATA_DIR, "packages.txt"), "r", encoding="utf-8") as f:
    full_text = f.read()

  if tier.lower() == "luxury":
    match = re.search(r"Luxury Tier:\s*\$([0-9]+)", full_text, re.IGNORECASE)
  else:
    match = re.search(r"Standard Tier:\s*\$([0-9]+)", full_text, re.IGNORECASE)

  if not match:
    raise ValueError("Package price not found in brochure.")

  return int(match.group(1))

def answer_user_query(user_query: str):

  selected_flight = select_flight(user_query)

  vectorstore = get_vectorstore()
  retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
  retrieved_docs = retriever.invoke(user_query)

  rag_context = "\n".join(doc.page_content for doc in retrieved_docs)

  people = _extract_people(user_query)
  package_tier = _extract_package_tier(user_query)

  flight_price_per_person = selected_flight["price_usd"]
  package_price_per_person = extract_package_price_with_fallback(
    rag_context, package_tier
  )

  flight_total = flight_price_per_person * people
  package_total = package_price_per_person * people
  total_cost = flight_total + package_total

  llm = get_llm()

  prompt = f"""
You are a professional travel assistant.

STRICT RULES (DO NOT BREAK):
- Do NOT perform calculations
- Do NOT modify prices
- Do NOT omit any line
- Do NOT suggest alternative packages
- Do NOT mention any destination other than the selected one
- You MUST repeat all prices exactly as provided

User query:
{user_query}

=====================
BOOKING SUMMARY
=====================

Flight:
- Airline: {selected_flight['airline']}
- Departure: {selected_flight['departure']}
- Price per person: ${flight_price_per_person}
- Flight total for {people} people: ${flight_total}

Package:
- Tier: {package_tier}
- Package price per person: ${package_price_per_person}
- Package total for {people} people: ${package_total}

FINAL TOTAL COST: ${total_cost}

=====================
PACKAGE INCLUSIONS
=====================
Use ONLY the information below. Do NOT add anything new.

{rag_context}

=====================
TASK
=====================
Explain the booking in clear language.
List ONLY the inclusions.
Do NOT repeat calculations.
"""

  return llm.invoke(prompt)