import sys
import os
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

from src.flight_logic import select_flight
from src.embeddings import load_embedding_model
from src.llm_setup import load_llm
from src.config import VECTOR_DB_DIR, TOP_K 


def _extract_people(query: str) -> int:
  match = re.search(r"(\d+)\s*(people|persons|travellers|travelers)", query.lower())
  return int(match.group(1)) if match else 1


def _extract_package_tier(query: str) -> str:
  return "Luxury" if "luxury" in query.lower() else "Standard"


def extract_package_price_from_context(context: str, tier: str) -> int:
  pricing_match = re.search(
  r"Pricing:(.*?)(?:Note:|$)",
  context,
  re.IGNORECASE | re.DOTALL
  )

  if not pricing_match:
    raise ValueError("Pricing section not found in context")

  pricing_text = pricing_match.group(1)

  tier_pattern = rf"{tier}\s+Package:\s*\$(\d+)\s+per\s+person"
  tier_match = re.search(tier_pattern, pricing_text, re.IGNORECASE)

  if tier_match:
    return int(tier_match.group(1))

  raise ValueError(f"{tier} price not found in pricing section")

def get_vectorstore():
  embeddings = load_embedding_model()
  vectorstore = FAISS.load_local(
      VECTOR_DB_DIR, 
      embeddings,
      allow_dangerous_deserialization=True
  )
  return vectorstore


def build_chain(retriever, llm, flight_data: dict, num_people: int, package_tier: str):
    
  def format_docs(docs):
    pricing_docs = [
        d.page_content for d in docs
        if "pricing:" in d.page_content.lower()
    ]

    if pricing_docs:
        return "\n\n".join(pricing_docs)

    return "\n\n".join(d.page_content for d in docs)

    
  def extract_and_calculate(inputs):
    context = inputs['context']
    question = inputs['question']

    package_price_per_person = extract_package_price_from_context(context, package_tier)

    flight_price_per_person = flight_data['price_usd']

    flight_total = flight_price_per_person * num_people
    package_total = package_price_per_person * num_people
    total_cost = flight_total + package_total

    return {
            "context": context,
            "question": question,
            "flight_id": flight_data['flight_id'],
            "airline": flight_data['airline'],
            "departure": flight_data['departure'],
            "flight_price_per_person": flight_price_per_person,
            "package_tier": package_tier,
            "package_price_per_person": package_price_per_person,
            "num_people": num_people,
            "flight_total": flight_total,
            "package_total": package_total,
            "total_cost": total_cost
        }
    
  parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    })
    
  calculation_chain = parallel_chain | RunnableLambda(extract_and_calculate)
    
  prompt = PromptTemplate(
        template="""You are a professional travel assistant.

User Query: {question}

BOOKING SUMMARY

FLIGHT DETAILS:
   • Airline: {airline} (Flight {flight_id})
   • Departure Time: {departure}
   • Price per person: ${flight_price_per_person}
   • Total for {num_people} people: ${flight_total}

PACKAGE DETAILS:
   • Tier: {package_tier}
   • Price per person: ${package_price_per_person}
   • Total for {num_people} people: ${package_total}

FINAL TOTAL COST: ${total_cost}

PACKAGE INCLUSIONS (from retrieved context):
{context}

INSTRUCTIONS

Provide a clear, professional summary of this booking in natural language.

Requirements:
1. Summarize the flight and package details
2. List ONLY the package inclusions that belong to the selected destination and tier
3. IGNORE any other destinations, packages, add-ons, or pricing mentioned in the context
4. Present the cost breakdown clearly
5. Do NOT modify any prices - use them exactly as provided
6. Do NOT perform calculations - they are already done
7. Do NOT add any signature, closing line, or role
8. Format the response in a clean, easy-to-read manner

Your response:
""",
        input_variables=[
            'question', 'context', 'flight_id', 'airline', 'departure',
            'flight_price_per_person', 'package_tier', 'package_price_per_person',
            'num_people', 'flight_total', 'package_total', 'total_cost'
        ]
    )
    
  parser = StrOutputParser()
    
  chain = calculation_chain | prompt | llm | parser
    
  return chain


def answer_user_query(user_query: str):
    print(f"\nUser Query: {user_query}\n")
    
    num_people = _extract_people(user_query)
    package_tier = _extract_package_tier(user_query)
    
    print(f"   • Number of people: {num_people}")
    print(f"   • Package tier: {package_tier}")
    
    flight_data = select_flight(user_query)
    
    if not flight_data:
        return "No flights found matching your criteria."
    
    print(f"   {flight_data['airline']} ({flight_data['flight_id']})")
    print(f"   Departure: {flight_data['departure']}")
    print(f"   Price: ${flight_data['price_usd']} per person")

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K })
    
    llm = load_llm()

    chain = build_chain(retriever, llm, flight_data, num_people, package_tier)
    
    result = chain.invoke(user_query)
    
    return result