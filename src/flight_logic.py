import json
from datetime import datetime

def _parse_time(time_str: str) -> int:
  return datetime.strptime(time_str, "%I:%M %p").hour

def select_flight(user_query: str) -> dict:
  with open("data/flights.json", "r") as f:
    flights = json.load(f)

  query = user_query.lower()

  if "morning" in query:
    flights = [f for f in flights if _parse_time(f["departure"]) < 12]
  elif "evening" in query or "night" in query:
    flights = [f for f in flights if _parse_time(f["departure"]) >= 17]

  flights.sort(key=lambda x: x["price_usd"])

  return flights[0]