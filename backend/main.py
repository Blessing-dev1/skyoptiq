import os
import json
from typing import Optional
from math import radians, sin, cos, sqrt, atan2

import serpapi
import httpx
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

app = FastAPI(title="SkyOpt IQ Backend", version="4.0.0")

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
    return {"message": "SkyOpt IQ backend is running", "version": "4.0.0"}


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


# ============================================================
# AIRPORT RESOLUTION
# ============================================================

# Practical commercial airports used for geographic origin discovery.
# Add more airports here over time, or replace this with a full airport DB.
AIRPORTS = [
    {"iata": "BOS", "name": "Boston Logan International", "city": "Boston, MA", "lat": 42.3656, "lon": -71.0096},
    {"iata": "MHT", "name": "Manchester-Boston Regional", "city": "Manchester, NH", "lat": 42.9326, "lon": -71.4357},
    {"iata": "BDL", "name": "Bradley International", "city": "Hartford/Springfield", "lat": 41.9389, "lon": -72.6832},
    {"iata": "ALB", "name": "Albany International", "city": "Albany, NY", "lat": 42.7483, "lon": -73.8017},
    {"iata": "PWM", "name": "Portland International Jetport", "city": "Portland, ME", "lat": 43.6462, "lon": -70.3093},
    {"iata": "BTV", "name": "Patrick Leahy Burlington International", "city": "Burlington, VT", "lat": 44.4719, "lon": -73.1533},
    {"iata": "JFK", "name": "John F. Kennedy International", "city": "New York, NY", "lat": 40.6413, "lon": -73.7781},
    {"iata": "EWR", "name": "Newark Liberty International", "city": "Newark, NJ", "lat": 40.6895, "lon": -74.1745},
    {"iata": "LGA", "name": "LaGuardia", "city": "New York, NY", "lat": 40.7769, "lon": -73.8740},
]

# Deterministic destination mapping for common cities. OpenAI is only used as
# a fallback/discovery planner, never as the geographic source for the origin.
DESTINATION_AIRPORTS = {
    "barcelona": [{"iata": "BCN", "city": "Barcelona", "reason": "Barcelona's primary international airport"}],
    "paris": [
        {"iata": "CDG", "city": "Paris", "reason": "Paris's primary international airport"},
        {"iata": "ORY", "city": "Paris", "reason": "Major secondary Paris airport"},
    ],
    "lisbon": [{"iata": "LIS", "city": "Lisbon", "reason": "Lisbon's primary airport"}],
    "malaga": [{"iata": "AGP", "city": "Malaga", "reason": "Malaga-Costa del Sol Airport"}],
    "madrid": [{"iata": "MAD", "city": "Madrid", "reason": "Madrid's primary international airport"}],
    "rome": [{"iata": "FCO", "city": "Rome", "reason": "Rome's primary international airport"}],
    "london": [
        {"iata": "LHR", "city": "London", "reason": "London Heathrow"},
        {"iata": "LGW", "city": "London", "reason": "London Gatwick"},
    ],
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.7613
    p1, p2 = radians(lat1), radians(lat2)
    dp = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return r * 2 * atan2(sqrt(a), sqrt(1 - a))


def geocode_origin(state: TripState) -> dict:
    """Resolve ZIP/city to coordinates using public Nominatim geocoding."""
    query = state.origin_zip or state.origin_text
    if not query:
        raise RuntimeError("Origin is missing")
    if state.origin_zip:
        query = f"{state.origin_zip}, USA"

    headers = {"User-Agent": "SkyOpt-IQ/4.0 (flight search application)"}
    params = {"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "us"}
    with httpx.Client(timeout=12.0, headers=headers) as client:
        response = client.get("https://nominatim.openstreetmap.org/search", params=params)
        response.raise_for_status()
        data = response.json()
    if not data:
        raise RuntimeError(f"Could not locate origin: {query}")
    return {
        "query": query,
        "display_name": data[0].get("display_name", query),
        "lat": float(data[0]["lat"]),
        "lon": float(data[0]["lon"]),
    }


def nearby_airports(state: TripState, max_results: int = 4) -> tuple[dict, list[dict]]:
    geo = geocode_origin(state)
    requested_radius = state.airport_radius_miles or 100
    # Use exactly the user's radius when supplied. Without one, use a practical
    # 100-mile default and report that fact in diagnostics.
    candidates = []
    for airport in AIRPORTS:
        miles = haversine_miles(geo["lat"], geo["lon"], airport["lat"], airport["lon"])
        if miles <= requested_radius:
            candidates.append({**airport, "distance_miles": round(miles, 1)})
    candidates.sort(key=lambda a: a["distance_miles"])
    return geo, candidates[:max_results]


SEARCH_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "destination_airports": {
            "type": "array", "maxItems": 4,
            "items": {"type": "object", "properties": {
                "iata": {"type": "string"}, "city": {"type": "string"}, "reason": {"type": "string"}
            }, "required": ["iata", "city", "reason"], "additionalProperties": False}
        },
        "summary": {"type": "string"}
    },
    "required": ["destination_airports", "summary"],
    "additionalProperties": False,
}


def resolve_destinations(state: TripState) -> dict:
    city_key = (state.destination_city or "").strip().lower()
    if city_key in DESTINATION_AIRPORTS:
        airports = DESTINATION_AIRPORTS[city_key]
        return {"destination_airports": airports, "summary": f"Resolved {state.destination_city} to {', '.join(a['iata'] for a in airports)}."}

    if not openai_client:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is missing")

    prompt = f"""
Create a compact destination-airport plan for this trip state:
{state.model_dump_json(indent=2)}

Rules:
- Return real commercial passenger-airport IATA codes only.
- If a destination city is specified, use airports that actually serve that city/metro.
- For discovery such as 'warm Europe', choose at most 4 sensible candidate cities.
- Do not invent prices or availability.
"""
    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        instructions="Resolve destination intent into a compact list of real destination airport IATA codes for flight search.",
        input=prompt,
        text={"format": {"type": "json_schema", "name": "skyopt_destination_plan", "strict": True, "schema": SEARCH_PLAN_SCHEMA}},
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
        geo, origin_airports = nearby_airports(state)
        if not origin_airports:
            radius = state.airport_radius_miles or 100
            return {
                "message": f"I located your origin, but no supported commercial airports were found within {radius} miles.",
                "resolved_origin": geo,
                "origin_airports": [],
                "destination_airports": [],
                "searched_pairs": [],
                "searches_attempted": 0,
                "flights_before_filters": 0,
                "flights": [],
                "errors": [],
            }

        destination_plan = resolve_destinations(state)
        destinations = destination_plan["destination_airports"][:4]
        all_flights = []
        searched_pairs = []
        errors = []
        attempts = 0

        # Limit API usage: at most 3 closest origin airports and 2 destinations
        # for a specific city; discovery requests can use up to 4 destinations.
        origin_limit = origin_airports[:3]
        dest_limit = destinations[:2] if state.destination_city else destinations[:4]

        for origin in origin_limit:
            for destination in dest_limit:
                o, d = origin["iata"].upper(), destination["iata"].upper()
                attempts += 1
                try:
                    flights = serp_search(
                        o, d, state.date_start, state.date_end,
                        state.cabin or "economy", state.passengers or 1,
                    )
                    searched_pairs.append({"origin": o, "destination": d, "results": len(flights)})
                    for f in flights:
                        f["origin_iata"] = o
                        f["origin_distance_miles"] = origin["distance_miles"]
                        f["destination_iata"] = d
                        f["destination_city"] = destination["city"]
                        f["destination_reason"] = destination["reason"]
                        all_flights.append(f)
                except Exception as exc:
                    errors.append({"pair": f"{o}-{d}", "error": str(exc)})

        flights_before_filters = len(all_flights)

        if state.max_stops is not None:
            all_flights = [f for f in all_flights if f.get("stops", 99) <= state.max_stops]

        # Only enforce budget after collecting real flights. This makes it clear
        # whether flights existed but were over budget versus no API results.
        over_budget_count = 0
        if state.budget_usd is not None:
            over_budget_count = sum(1 for f in all_flights if (f.get("price") or 0) > state.budget_usd)
            all_flights = [f for f in all_flights if 0 < (f.get("price") or 0) <= state.budget_usd]

        seen, deduped = set(), []
        for f in sorted(all_flights, key=lambda x: (x.get("price") or 10**9, x.get("duration") or 10**9)):
            key = (f.get("origin_iata"), f.get("destination_iata"), f.get("flight_code"), f.get("price"), f.get("dep_time"))
            if key not in seen:
                seen.add(key)
                deduped.append(f)

        radius = state.airport_radius_miles or 100
        origin_codes = [a["iata"] for a in origin_limit]
        dest_codes = [a["iata"] for a in dest_limit]
        if deduped:
            message = f"Searched {', '.join(origin_codes)} to {', '.join(dest_codes)} and found {len(deduped)} matching options."
        elif flights_before_filters and state.budget_usd is not None:
            message = f"I found {flights_before_filters} flight options, but none remained after your current filters/budget of ${state.budget_usd:,.0f}."
        else:
            message = f"I searched {', '.join(origin_codes)} to {', '.join(dest_codes)} but the flight provider returned no matching itineraries for these dates."

        return {
            "message": message,
            "resolved_origin": geo,
            "airport_radius_miles": radius,
            "origin_airports": origin_limit,
            "destination_airports": dest_limit,
            "searched_pairs": searched_pairs,
            "searches_attempted": attempts,
            "flights_before_filters": flights_before_filters,
            "over_budget_count": over_budget_count,
            "flights": deduped[:100],
            "errors": errors[:10],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI flight search error: {str(e)}")

