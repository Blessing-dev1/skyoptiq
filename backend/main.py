import os
import json
from typing import Optional

import serpapi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
FRONTEND_URL = os.getenv("FRONTEND_URL")

app = FastAPI(title="SkyOpt IQ Backend", version="3.0.0")

allowed_origins = ["http://127.0.0.1:5500", "http://localhost:5500"]
if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class FlightSearchRequest(BaseModel):
    trip_type: str = "oneway"
    origin: str
    destination: str
    depart_date: str
    return_date: Optional[str] = None
    second_depart_date: Optional[str] = None
    cabin: str = "economy"
    passengers: int = 1
    weight_price: float = 0.5
    weight_stops: float = 0.3
    weight_duration: float = 0.2


class TripState(BaseModel):
    origin_text: Optional[str] = None
    origin_zip: Optional[str] = None
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    destination_region: Optional[str] = None
    destination_preferences: list[str] = Field(default_factory=list)
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    budget_usd: Optional[float] = None
    budget_is_per_person: Optional[bool] = None
    passengers: Optional[int] = None
    cabin: Optional[str] = None
    max_stops: Optional[int] = None
    checked_bags: Optional[int] = None
    airport_radius_miles: Optional[int] = None


class AIChatRequest(BaseModel):
    message: str
    state: TripState = Field(default_factory=TripState)


class AIChatResponse(BaseModel):
    message: str
    state: TripState
    ready_to_search: bool
    missing_fields: list[str] = Field(default_factory=list)


class AISearchRequest(BaseModel):
    state: TripState


@app.get("/")
def root():
    return {"message": "SkyOpt IQ backend is running", "version": "3.0.0"}


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "serpapi_configured": bool(SERPAPI_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
    }


def get_travel_class(cabin: str) -> int:
    return {
        "economy": 1,
        "premium": 2,
        "premium economy": 2,
        "premium_economy": 2,
        "business": 3,
        "first": 4,
        "first class": 4,
    }.get((cabin or "economy").lower(), 1)


def normalize_flight(entry: dict) -> Optional[dict]:
    flights = entry.get("flights", [])
    if not flights:
        return None
    first_leg, last_leg = flights[0], flights[-1]
    return {
        "source": "serp",
        "price": entry.get("price", 0),
        "airline": first_leg.get("airline") or "Unknown",
        "logo": first_leg.get("airline_logo"),
        "stops": max(len(flights) - 1, 0),
        "duration": entry.get("total_duration", 0),
        "dep_time": first_leg.get("departure_airport", {}).get("time"),
        "arr_time": last_leg.get("arrival_airport", {}).get("time"),
        "flight_code": first_leg.get("flight_number"),
        "legs": flights,
        "layovers": entry.get("layovers", []),
        "extensions": entry.get("extensions", []),
        "fare_name": entry.get("fare_name"),
        "carbon_emissions": entry.get("carbon_emissions"),
    }


def serp_search(origin: str, destination: str, depart_date: str,
                return_date: Optional[str], cabin: str, passengers: int) -> list[dict]:
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY is missing")

    params = {
        "engine": "google_flights",
        "departure_id": origin.upper(),
        "arrival_id": destination.upper(),
        "outbound_date": depart_date,
        "currency": "USD",
        "hl": "en",
        "adults": max(passengers, 1),
        "travel_class": get_travel_class(cabin),
        "type": 1 if return_date else 2,
    }
    if return_date:
        params["return_date"] = return_date

    client = serpapi.Client(api_key=SERPAPI_KEY)
    results = client.search(params)
    if "error" in results:
        raise RuntimeError(results["error"])

    normalized = []
    for entry in results.get("best_flights", []) + results.get("other_flights", []):
        flight = normalize_flight(entry)
        if flight:
            normalized.append(flight)
    return normalized


@app.post("/api/v1/flights/search")
def search_flights(payload: FlightSearchRequest):
    trip_type = payload.trip_type.lower()
    if trip_type not in {"oneway", "roundtrip", "multi"}:
        raise HTTPException(status_code=400, detail="trip_type must be oneway, roundtrip, or multi")
    if trip_type == "roundtrip" and not payload.return_date:
        raise HTTPException(status_code=400, detail="return_date is required for roundtrip")
    if trip_type == "multi":
        raise HTTPException(status_code=400, detail="Multi-city search is not implemented yet")

    try:
        flights = serp_search(
            payload.origin, payload.destination, payload.depart_date,
            payload.return_date if trip_type == "roundtrip" else None,
            payload.cabin, payload.passengers,
        )
        return {"origin": payload.origin.upper(), "destination": payload.destination.upper(), "flights": flights}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flight search error: {str(e)}")


SKYOPT_INSTRUCTIONS = """
You are SkyOpt IQ, a conversational travel-search assistant. Build and preserve a structured trip state.
Never ask users for airport codes. Accept city/town/state/ZIP origins. If origin is only broad (for example USA), ask for city or ZIP.
A specific city such as Boston or Keene, NH is sufficient; do not ask for ZIP after that.
Accept specific destinations or discovery requests such as 'somewhere warm in Europe'.
Ask only ONE important missing question at a time and never ask again for information already in state.
Dates must be YYYY-MM-DD. Current date is 2026-09-21. If the user gives Dec 23-Dec 31 without a year, use 2026.
If they only say Christmas and exact dates are absent, ask for their date range.
Never invent flight prices, schedules, or availability. Actual flight data comes from SerpAPI.
Set ready_to_search=true when origin, usable dates, and either a destination city/country/region are known.
"""

TRIP_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string"},
        "state": {
            "type": "object",
            "properties": {
                "origin_text": {"type": ["string", "null"]},
                "origin_zip": {"type": ["string", "null"]},
                "destination_city": {"type": ["string", "null"]},
                "destination_country": {"type": ["string", "null"]},
                "destination_region": {"type": ["string", "null"]},
                "destination_preferences": {"type": "array", "items": {"type": "string"}},
                "date_start": {"type": ["string", "null"]},
                "date_end": {"type": ["string", "null"]},
                "budget_usd": {"type": ["number", "null"]},
                "budget_is_per_person": {"type": ["boolean", "null"]},
                "passengers": {"type": ["integer", "null"]},
                "cabin": {"type": ["string", "null"]},
                "max_stops": {"type": ["integer", "null"]},
                "checked_bags": {"type": ["integer", "null"]},
                "airport_radius_miles": {"type": ["integer", "null"]},
            },
            "required": ["origin_text", "origin_zip", "destination_city", "destination_country", "destination_region",
                         "destination_preferences", "date_start", "date_end", "budget_usd", "budget_is_per_person",
                         "passengers", "cabin", "max_stops", "checked_bags", "airport_radius_miles"],
            "additionalProperties": False,
        },
        "ready_to_search": {"type": "boolean"},
        "missing_fields": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["message", "state", "ready_to_search", "missing_fields"],
    "additionalProperties": False,
}


@app.post("/api/v1/ai/chat", response_model=AIChatResponse)
def ai_chat(payload: AIChatRequest):
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="message cannot be empty")

    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=SKYOPT_INSTRUCTIONS,
            input=f"CURRENT TRIP STATE:\n{payload.state.model_dump_json(indent=2)}\n\nNEW USER MESSAGE:\n{payload.message}",
            text={"format": {"type": "json_schema", "name": "skyopt_trip_state", "strict": True, "schema": TRIP_SCHEMA}},
        )
        return AIChatResponse(**json.loads(response.output_text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SkyOpt AI error: {str(e)}")


SEARCH_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "origin_airports": {
            "type": "array", "maxItems": 2,
            "items": {"type": "object", "properties": {
                "iata": {"type": "string"}, "label": {"type": "string"}
            }, "required": ["iata", "label"], "additionalProperties": False}
        },
        "destination_airports": {
            "type": "array", "maxItems": 4,
            "items": {"type": "object", "properties": {
                "iata": {"type": "string"}, "city": {"type": "string"}, "reason": {"type": "string"}
            }, "required": ["iata", "city", "reason"], "additionalProperties": False}
        },
        "summary": {"type": "string"}
    },
    "required": ["origin_airports", "destination_airports", "summary"],
    "additionalProperties": False,
}


def build_search_plan(state: TripState) -> dict:
    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")

    prompt = f"""
Create a small airport search plan for this trip state:
{state.model_dump_json(indent=2)}

Rules:
- Return real commercial passenger-airport IATA codes only.
- For a specific origin city, prefer the primary practical airport. For a smaller city/town, you may include up to 2 practical airports.
- For a specific destination city, include at most 2 practical airports serving that city/metro.
- For a discovery destination such as 'warm Europe', choose at most 4 candidate cities that fit the stated preferences and season.
- Do not invent prices or availability.
- Keep the plan small because each origin/destination pair consumes a flight API search.
"""
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions="You convert a completed travel intent into a compact list of real airport IATA codes for downstream flight search.",
        input=prompt,
        text={"format": {"type": "json_schema", "name": "skyopt_search_plan", "strict": True, "schema": SEARCH_PLAN_SCHEMA}},
    )
    return json.loads(response.output_text)


@app.post("/api/v1/ai/search")
def ai_search(payload: AISearchRequest):
    state = payload.state
    if not state.date_start:
        raise HTTPException(status_code=400, detail="Trip dates are missing")
    if not (state.origin_text or state.origin_zip):
        raise HTTPException(status_code=400, detail="Origin is missing")
    if not (state.destination_city or state.destination_country or state.destination_region):
        raise HTTPException(status_code=400, detail="Destination is missing")

    try:
        plan = build_search_plan(state)
        all_flights = []
        searched_pairs = []
        errors = []

        for origin in plan["origin_airports"][:2]:
            for destination in plan["destination_airports"][:4]:
                o, d = origin["iata"].upper(), destination["iata"].upper()
                try:
                    flights = serp_search(
                        o, d, state.date_start, state.date_end,
                        state.cabin or "economy", state.passengers or 1,
                    )
                    searched_pairs.append(f"{o}-{d}")
                    for f in flights:
                        f["origin_iata"] = o
                        f["destination_iata"] = d
                        f["destination_city"] = destination["city"]
                        f["destination_reason"] = destination["reason"]
                        all_flights.append(f)
                except Exception as exc:
                    errors.append(f"{o}-{d}: {str(exc)}")

        if state.max_stops is not None:
            all_flights = [f for f in all_flights if f.get("stops", 99) <= state.max_stops]
        if state.budget_usd is not None:
            all_flights = [f for f in all_flights if 0 < (f.get("price") or 0) <= state.budget_usd]

        # Remove obvious duplicates and sort cheaply before frontend applies user weights.
        seen, deduped = set(), []
        for f in sorted(all_flights, key=lambda x: (x.get("price") or 10**9, x.get("duration") or 10**9)):
            key = (f.get("origin_iata"), f.get("destination_iata"), f.get("flight_code"), f.get("price"), f.get("dep_time"))
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        return {
            "message": plan["summary"],
            "plan": plan,
            "searched_pairs": searched_pairs,
            "flights": deduped[:100],
            "errors": errors[:10],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI flight search error: {str(e)}")
