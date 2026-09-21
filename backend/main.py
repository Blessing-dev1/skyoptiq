import os
import json
from typing import Optional
from datetime import datetime, timezone
import secrets
from math import radians, sin, cos, sqrt, atan2

import serpapi
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
import psycopg
from psycopg.rows import dict_row

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
FRONTEND_URL = os.getenv("FRONTEND_URL")
DUFFEL_ACCESS_TOKEN = os.getenv("DUFFEL_ACCESS_TOKEN")
DUFFEL_BASE_URL = "https://api.duffel.com"
DATABASE_URL = os.getenv("DATABASE_URL")

app = FastAPI(title="SkyBas Backend", version="4.3.0")

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


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {
        "message": "SkyBas backend is running",
        "version": "4.3.0"
    }

@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "serpapi_configured": bool(SERPAPI_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
        "duffel_configured": bool(DUFFEL_ACCESS_TOKEN),
        "database_configured": bool(DATABASE_URL),
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
    except Exception:
        raise HTTPException(status_code=502, detail="Flight provider request failed.")


SKYOPT_INSTRUCTIONS = """
You are SkyBas, a conversational travel-search assistant. Build and preserve a structured trip state.
Never ask users for airport codes. Accept city/town/state/ZIP origins. If origin is only broad (for example USA), ask for city or ZIP.
A specific city such as Boston or Keene, NH is sufficient; do not ask for ZIP after that.
Accept specific destinations or discovery requests such as 'somewhere warm in Europe'.
Ask only ONE important missing question at a time and never ask again for information already in state.
Dates must be YYYY-MM-DD. Current date is 2026-09-21. If the user gives Dec 23-Dec 31 without a year, use 2026.
If they only say Christmas and exact dates are absent, ask for their date range.
Never invent flight prices, schedules, or availability. Actual flight data comes from SerpAPI.

Conversation rules:
- Preserve facts already present in CURRENT TRIP STATE unless the user explicitly changes them.
- passengers defaults to 1 and cabin defaults to economy when not otherwise specified.
- If the user supplies a budget, store it in budget_usd.
- When budget_usd is known but budget_is_per_person is null, ask whether the budget is TOTAL for the trip or PER PERSON.
- While that budget-scope clarification is unresolved, ready_to_search MUST be false and missing_fields MUST include "budget_is_per_person".
- "per person", "each", or an equivalent reply means budget_is_per_person=true.
- "total", "whole trip", "for everyone", or an equivalent reply means budget_is_per_person=false.
- A bare "yes", "ok", "okay", "okey", "sure", or similar acknowledgement does NOT answer a total-vs-per-person question. Ask the user to say "total" or "per person".
- If the user says something like "yes, one person" while a budget-scope clarification is pending, set passengers=1 but still ask "total or per person" unless they explicitly say per person/each or total/for everyone.
- Do not claim ready_to_search=true in the message when a required clarification is still missing.
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



def finalize_chat_state(previous: TripState, result: AIChatResponse, user_message: str) -> AIChatResponse:
    """
    Deterministic guardrail after the model response.
    The model extracts intent; this function decides whether a real search is allowed.
    """
    state = result.state

    # Product defaults: manual search and AI search should behave consistently.
    if state.passengers is None:
        state.passengers = previous.passengers or 1
    if not state.cabin:
        state.cabin = previous.cabin or "economy"

    # Preserve an already-known budget if the model accidentally drops it.
    if state.budget_usd is None and previous.budget_usd is not None:
        state.budget_usd = previous.budget_usd
    if state.budget_is_per_person is None and previous.budget_is_per_person is not None:
        state.budget_is_per_person = previous.budget_is_per_person

    text = (user_message or "").strip().lower()

    # Resolve explicit budget-scope replies deterministically.
    per_person_phrases = ("per person", "each person", "per traveler", "per traveller", "each traveler", "each traveller")
    total_phrases = ("total", "whole trip", "for everyone", "for everybody", "all passengers", "altogether")

    if state.budget_usd is not None:
        if any(p in text for p in per_person_phrases):
            state.budget_is_per_person = True
        elif any(p in text for p in total_phrases):
            state.budget_is_per_person = False

    missing = []
    if not (state.origin_text or state.origin_zip):
        missing.append("origin")
    if not state.date_start:
        missing.append("date_start")
    if not (state.destination_city or state.destination_country or state.destination_region):
        missing.append("destination")

    # A supplied budget has an explicit scope requirement.
    if state.budget_usd is not None and state.budget_is_per_person is None:
        missing.append("budget_is_per_person")

    ready = len(missing) == 0

    # Never let a model-generated "ready" message contradict deterministic readiness.
    if "budget_is_per_person" in missing:
        message = (
            f"Is your ${state.budget_usd:,.0f} budget total for the whole trip "
            "or per person?"
        )
    else:
        message = result.message

    return AIChatResponse(
        message=message,
        state=state,
        ready_to_search=ready,
        missing_fields=missing,
    )


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
        parsed = AIChatResponse(**json.loads(response.output_text))
        return finalize_chat_state(payload.state, parsed, payload.message)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="SkyBas AI request failed.")


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

    headers = {"User-Agent": "SkyBas/4.0 (flight search application)"}
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
                except Exception:
                    errors.append({"pair": f"{o}-{d}", "error": "Flight provider request failed."})

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
    except Exception:
        raise HTTPException(status_code=502, detail="AI flight search failed.")



# ============================================================
# SKYOPT BOOKING DATABASE
# ============================================================

def db_connect():
    if not DATABASE_URL:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    return psycopg.connect(DATABASE_URL, row_factory=dict_row)


def init_database():
    if not DATABASE_URL:
        return
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS skyopt_bookings (
                    id BIGSERIAL PRIMARY KEY,
                    manage_token TEXT NOT NULL UNIQUE,
                    provider TEXT NOT NULL DEFAULT 'duffel',
                    provider_order_id TEXT NOT NULL UNIQUE,
                    booking_reference TEXT,
                    status TEXT NOT NULL DEFAULT 'confirmed',
                    test_mode BOOLEAN NOT NULL DEFAULT TRUE,
                    total_amount NUMERIC(12,2),
                    total_currency VARCHAR(8),
                    contact_email TEXT,
                    airline TEXT,
                    origin_iata VARCHAR(8),
                    destination_iata VARCHAR(8),
                    departing_at TIMESTAMPTZ,
                    arriving_at TIMESTAMPTZ,
                    passenger_count INTEGER NOT NULL DEFAULT 1,
                    available_actions JSONB NOT NULL DEFAULT '[]'::jsonb,
                    itinerary JSONB NOT NULL DEFAULT '{}'::jsonb,
                    provider_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_skyopt_bookings_created_at ON skyopt_bookings(created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_skyopt_bookings_contact_email ON skyopt_bookings(contact_email)")
        conn.commit()


@app.on_event("startup")
def startup_database():
    init_database()


def _order_summary(order: dict) -> dict:
    slices = order.get("slices") or []
    segments = [seg for sl in slices for seg in (sl.get("segments") or [])]
    first = segments[0] if segments else {}
    last = segments[-1] if segments else {}
    owner = order.get("owner") or {}
    contact_email = None
    passengers = order.get("passengers") or []
    for p in passengers:
        if p.get("email"):
            contact_email = p.get("email")
            break
    return {
        "contact_email": contact_email,
        "airline": owner.get("name"),
        "origin_iata": ((slices[0].get("origin") or {}).get("iata_code") if slices else None),
        "destination_iata": ((slices[0].get("destination") or {}).get("iata_code") if slices else None),
        "departing_at": first.get("departing_at"),
        "arriving_at": last.get("arriving_at"),
        "passenger_count": len(passengers) or 1,
        "itinerary": {"slices": slices},
    }


def save_booking(order: dict, test_mode: bool = True) -> dict:
    summary = _order_summary(order)
    token = secrets.token_urlsafe(32)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO skyopt_bookings (
                    manage_token, provider_order_id, booking_reference, status, test_mode,
                    total_amount, total_currency, contact_email, airline, origin_iata,
                    destination_iata, departing_at, arriving_at, passenger_count,
                    available_actions, itinerary, provider_snapshot
                ) VALUES (
                    %(manage_token)s, %(provider_order_id)s, %(booking_reference)s, %(status)s, %(test_mode)s,
                    %(total_amount)s, %(total_currency)s, %(contact_email)s, %(airline)s, %(origin_iata)s,
                    %(destination_iata)s, %(departing_at)s, %(arriving_at)s, %(passenger_count)s,
                    %(available_actions)s::jsonb, %(itinerary)s::jsonb, %(provider_snapshot)s::jsonb
                )
                ON CONFLICT (provider_order_id) DO UPDATE SET
                    booking_reference=EXCLUDED.booking_reference,
                    status=EXCLUDED.status,
                    total_amount=EXCLUDED.total_amount,
                    total_currency=EXCLUDED.total_currency,
                    available_actions=EXCLUDED.available_actions,
                    itinerary=EXCLUDED.itinerary,
                    provider_snapshot=EXCLUDED.provider_snapshot,
                    updated_at=NOW()
                RETURNING id, manage_token, provider_order_id, booking_reference, status, test_mode,
                          total_amount, total_currency, airline, origin_iata, destination_iata,
                          departing_at, arriving_at, passenger_count, created_at, updated_at
            """, {
                "manage_token": token,
                "provider_order_id": order.get("id"),
                "booking_reference": order.get("booking_reference"),
                "status": "confirmed",
                "test_mode": test_mode,
                "total_amount": order.get("total_amount"),
                "total_currency": order.get("total_currency"),
                **summary,
                "available_actions": json.dumps(order.get("available_actions") or []),
                "itinerary": json.dumps(summary["itinerary"]),
                "provider_snapshot": json.dumps(order),
            })
            row = cur.fetchone()
        conn.commit()
    return dict(row)


def _require_booking_access(booking_id: int, manage_token: str):
    if not manage_token:
        raise HTTPException(status_code=401, detail="Booking management token is required")
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, manage_token, provider_order_id, booking_reference, status, test_mode,
                       total_amount, total_currency, airline, origin_iata, destination_iata,
                       departing_at, arriving_at, passenger_count, available_actions,
                       itinerary, created_at, updated_at
                FROM skyopt_bookings
                WHERE id=%s AND manage_token=%s
            """, (booking_id, manage_token))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Booking not found")
    return dict(row)


@app.get("/api/v1/bookings/{booking_id}")
def get_booking(booking_id: int, manage_token: str):
    return {"booking": _require_booking_access(booking_id, manage_token)}


@app.get("/api/v1/bookings/{booking_id}/refresh")
def refresh_booking(booking_id: int, manage_token: str):
    booking = _require_booking_access(booking_id, manage_token)
    order_id = booking["provider_order_id"]
    with httpx.Client(timeout=25.0) as client:
        response = client.get(f"{DUFFEL_BASE_URL}/air/orders/{order_id}", headers=duffel_headers())
    if response.status_code >= 400:
        raise HTTPException(status_code=502, detail=duffel_error_detail(response))
    order = response.json().get("data") or {}
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE skyopt_bookings SET
                    booking_reference=%s, total_amount=%s, total_currency=%s,
                    available_actions=%s::jsonb, provider_snapshot=%s::jsonb, updated_at=NOW()
                WHERE id=%s AND manage_token=%s
            """, (
                order.get("booking_reference"), order.get("total_amount"), order.get("total_currency"),
                json.dumps(order.get("available_actions") or []), json.dumps(order), booking_id, manage_token
            ))
        conn.commit()
    return {"booking": _require_booking_access(booking_id, manage_token), "provider_order": order}


# ============================================================
# DUFFEL TRANSACTIONAL FLIGHTS (TEST MODE FIRST)
# ============================================================

class DuffelSearchRequest(BaseModel):
    origin: str
    destination: str
    depart_date: str
    return_date: Optional[str] = None
    cabin: str = "economy"
    passengers: int = Field(default=1, ge=1, le=9)
    max_connections: Optional[int] = Field(default=1, ge=0, le=3)


class DuffelPassenger(BaseModel):
    id: str
    given_name: str
    family_name: str
    born_on: str
    gender: str
    title: str
    email: Optional[str] = None
    phone_number: Optional[str] = None


class DuffelTestOrderRequest(BaseModel):
    offer_id: str
    passengers: list[DuffelPassenger]


def duffel_headers():
    if not DUFFEL_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="DUFFEL_ACCESS_TOKEN is not configured")
    return {
        "Authorization": f"Bearer {DUFFEL_ACCESS_TOKEN}",
        "Duffel-Version": "v2",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def duffel_cabin(cabin: str) -> str:
    value = (cabin or "economy").strip().lower().replace(" ", "_")
    if value == "premium":
        value = "premium_economy"
    if value not in {"economy", "premium_economy", "business", "first"}:
        value = "economy"
    return value


def _duration_minutes(iso_duration: Optional[str]) -> int:
    if not iso_duration:
        return 0
    # Duffel durations are ISO-8601 strings such as PT7H30M.
    import re
    m = re.fullmatch(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?", iso_duration)
    if not m:
        return 0
    days, hours, minutes = (int(x or 0) for x in m.groups())
    return days * 1440 + hours * 60 + minutes


def normalize_duffel_offer(offer: dict) -> dict:
    slices = offer.get("slices") or []
    segments = [seg for sl in slices for seg in (sl.get("segments") or [])]
    first = segments[0] if segments else {}
    last = segments[-1] if segments else {}

    owner = offer.get("owner") or {}
    operating_names = []
    for seg in segments:
        name = (seg.get("operating_carrier") or {}).get("name")
        if name and name not in operating_names:
            operating_names.append(name)

    total_connections = sum(max(len(sl.get("segments") or []) - 1, 0) for sl in slices)
    total_duration = sum(_duration_minutes(sl.get("duration")) for sl in slices)

    origin = ((slices[0].get("origin") or {}).get("iata_code") if slices else None)
    destination = ((slices[0].get("destination") or {}).get("iata_code") if slices else None)

    return {
        "source": "duffel",
        "bookable": True,
        "offer_id": offer.get("id"),
        "expires_at": offer.get("expires_at"),
        "live_mode": offer.get("live_mode"),
        "price": float(offer.get("total_amount") or 0),
        "currency": offer.get("total_currency"),
        "airline": owner.get("name") or (operating_names[0] if operating_names else "Unknown"),
        "operating_carriers": operating_names,
        "logo": owner.get("logo_symbol_url") or owner.get("logo_lockup_url"),
        "stops": total_connections,
        "duration": total_duration,
        "dep_time": first.get("departing_at"),
        "arr_time": last.get("arriving_at"),
        "flight_code": (
            f"{(first.get('marketing_carrier') or {}).get('iata_code', '')}"
            f"{first.get('marketing_carrier_flight_number') or ''}"
        ).strip(),
        "origin_iata": origin,
        "destination_iata": destination,
        "payment_requirements": offer.get("payment_requirements"),
        "conditions": offer.get("conditions"),
        "passengers": offer.get("passengers") or [],
        "slices": slices,
    }


def duffel_error_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
        errors = payload.get("errors") or []
        if errors:
            return errors[0].get("message") or errors[0].get("title") or "Duffel request failed"
    except Exception:
        pass
    return "Duffel request failed"


@app.get("/api/v1/duffel/health")
def duffel_health():
    if not DUFFEL_ACCESS_TOKEN:
        raise HTTPException(status_code=500, detail="DUFFEL_ACCESS_TOKEN is not configured")
    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.get(
                f"{DUFFEL_BASE_URL}/air/airlines",
                headers=duffel_headers(),
                params={"limit": 1},
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail="Duffel connection failed")
        return {
            "status": "ok",
            "duffel_configured": True,
            "duffel_connected": True,
            "mode": "test" if DUFFEL_ACCESS_TOKEN.startswith("duffel_test_") else "live",
            "api_version": "v2",
            "message": "SkyBas is connected to Duffel",
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Duffel connection failed")


@app.post("/api/v1/duffel/search")
def duffel_search(payload: DuffelSearchRequest):
    slices = [{
        "origin": payload.origin.upper(),
        "destination": payload.destination.upper(),
        "departure_date": payload.depart_date,
    }]
    if payload.return_date:
        slices.append({
            "origin": payload.destination.upper(),
            "destination": payload.origin.upper(),
            "departure_date": payload.return_date,
        })

    body = {
        "data": {
            "slices": slices,
            "passengers": [{"type": "adult"} for _ in range(payload.passengers)],
            "cabin_class": duffel_cabin(payload.cabin),
            "max_connections": payload.max_connections,
        }
    }

    try:
        # Supplier timeout is lower than the HTTP timeout so Duffel can return
        # completed airline results before our request itself times out.
        with httpx.Client(timeout=35.0) as client:
            response = client.post(
                f"{DUFFEL_BASE_URL}/air/offer_requests",
                headers=duffel_headers(),
                params={"return_offers": "true", "supplier_timeout": 20000},
                json=body,
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=duffel_error_detail(response))

        data = response.json().get("data") or {}
        offers = [normalize_duffel_offer(x) for x in (data.get("offers") or [])]
        offers.sort(key=lambda x: (x.get("price") or 10**9, x.get("stops") or 0, x.get("duration") or 10**9))
        return {
            "offer_request_id": data.get("id"),
            "live_mode": data.get("live_mode"),
            "origin": payload.origin.upper(),
            "destination": payload.destination.upper(),
            "flights": offers[:100],
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Duffel flight search failed.")


@app.get("/api/v1/duffel/offers/{offer_id}")
def duffel_get_offer(offer_id: str):
    if not offer_id.startswith("off_"):
        raise HTTPException(status_code=400, detail="Invalid Duffel offer ID")
    try:
        with httpx.Client(timeout=25.0) as client:
            response = client.get(
                f"{DUFFEL_BASE_URL}/air/offers/{offer_id}",
                headers=duffel_headers(),
            )
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Offer was not found or is no longer available")
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=duffel_error_detail(response))
        offer = response.json().get("data") or {}
        return {"offer": normalize_duffel_offer(offer)}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Could not refresh Duffel offer.")


@app.post("/api/v1/duffel/orders/test")
def duffel_create_test_order(payload: DuffelTestOrderRequest):
    # Hard safety rail: this endpoint must never create a live booking.
    if not DUFFEL_ACCESS_TOKEN or not DUFFEL_ACCESS_TOKEN.startswith("duffel_test_"):
        raise HTTPException(status_code=403, detail="Test booking endpoint requires a Duffel test token")

    # Refresh the offer so checkout uses the current amount/currency and so we
    # fail before order creation if the offer has expired.
    try:
        with httpx.Client(timeout=25.0) as client:
            offer_response = client.get(
                f"{DUFFEL_BASE_URL}/air/offers/{payload.offer_id}",
                headers=duffel_headers(),
            )
        if offer_response.status_code >= 400:
            raise HTTPException(status_code=409, detail="Selected offer is no longer available. Search again.")
        offer = offer_response.json().get("data") or {}

        offer_passengers = offer.get("passengers") or []
        supplied = {p.id: p for p in payload.passengers}
        if len(supplied) != len(offer_passengers):
            raise HTTPException(status_code=400, detail="Passenger count does not match the selected offer")

        passengers = []
        for op in offer_passengers:
            pid = op.get("id")
            p = supplied.get(pid)
            if not p:
                raise HTTPException(status_code=400, detail=f"Missing passenger details for {pid}")
            item = {
                "id": pid,
                "given_name": p.given_name,
                "family_name": p.family_name,
                "born_on": p.born_on,
                "gender": p.gender,
                "title": p.title,
            }
            if p.email:
                item["email"] = p.email
            if p.phone_number:
                item["phone_number"] = p.phone_number
            passengers.append(item)

        amount = offer.get("total_amount")
        currency = offer.get("total_currency")
        if not amount or not currency:
            raise HTTPException(status_code=409, detail="Selected offer does not contain a valid total price")

        order_body = {
            "data": {
                "type": "instant",
                "selected_offers": [payload.offer_id],
                "passengers": passengers,
                "payments": [{
                    "type": "balance",
                    "amount": amount,
                    "currency": currency,
                }],
            }
        }

        # Booking calls can take much longer than ordinary API requests.
        with httpx.Client(timeout=140.0) as client:
            response = client.post(
                f"{DUFFEL_BASE_URL}/air/orders",
                headers=duffel_headers(),
                json=order_body,
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=duffel_error_detail(response))

        order = response.json().get("data") or {}
        booking = save_booking(order, test_mode=True)
        return {
            "status": "confirmed",
            "test_mode": True,
            "booking_id": booking.get("id"),
            "manage_token": booking.get("manage_token"),
            "order_id": order.get("id"),
            "booking_reference": order.get("booking_reference"),
            "total_amount": order.get("total_amount"),
            "total_currency": order.get("total_currency"),
            "available_actions": order.get("available_actions") or [],
            "order": order,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Duffel test booking failed.")


@app.get("/api/v1/duffel/orders/{order_id}")
def duffel_get_order(order_id: str):
    if not order_id.startswith("ord_"):
        raise HTTPException(status_code=400, detail="Invalid Duffel order ID")
    try:
        with httpx.Client(timeout=25.0) as client:
            response = client.get(
                f"{DUFFEL_BASE_URL}/air/orders/{order_id}",
                headers=duffel_headers(),
            )
        if response.status_code >= 400:
            raise HTTPException(status_code=502, detail=duffel_error_detail(response))
        order = response.json().get("data") or {}
        return {
            "order_id": order.get("id"),
            "booking_reference": order.get("booking_reference"),
            "total_amount": order.get("total_amount"),
            "total_currency": order.get("total_currency"),
            "available_actions": order.get("available_actions") or [],
            "order": order,
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Could not retrieve Duffel order.")
