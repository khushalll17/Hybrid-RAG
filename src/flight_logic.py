import json
from datetime import datetime
from typing import List, Dict, Optional

def _parse_time(time_str: str) -> int:
  return datetime.strptime(time_str, "%I:%M %p").hour

def _parse_price(price: float) -> float:
  return float(price)

def select_flight(user_query: str) -> Optional[Dict]:
  with open("data/flights.json", "r") as f:
    flights = json.load(f)
    
  if not flights:
    return None
    
  query = user_query.lower()
  filtered_flights = flights.copy()
    
  if "morning" in query or "early" in query:
    filtered_flights = [
      f for f in filtered_flights 
      if 5 <= _parse_time(f["departure"]) < 12
    ]
    
  elif "afternoon" in query or "midday" in query or "noon" in query:
    filtered_flights = [
      f for f in filtered_flights 
      if 12 <= _parse_time(f["departure"]) < 17
    ]
    
  elif "evening" in query:
    filtered_flights = [
      f for f in filtered_flights 
      if 17 <= _parse_time(f["departure"]) < 21
    ]
    
  elif "night" in query or "late" in query:
    filtered_flights = [
      f for f in filtered_flights 
      if _parse_time(f["departure"]) >= 21 or _parse_time(f["departure"]) < 5
    ]
    
  if not filtered_flights:
    return None
    
  filtered_flights.sort(key=lambda x: _parse_price(x["price_usd"]))
    
  if any(word in query for word in ["expensive", "premium"]):
    return flights[-1]

  if any(word in query for word in ["mid", "mid range", "average"]):
    return flights[len(flights) // 2]

  return flights[0]