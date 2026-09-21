import os
import json
from typing import Optional

import serpapi
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
FRONTEND_URL = os.getenv("FRONTEND_URL")


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="SkyOpt IQ Backend",
    version="2.0.0",
    description="AI-powered flight search and travel optimization backend",
)


# ============================================================
# CORS
# ============================================================

allowed_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
]

if FRONTEND_URL:
    allowed_origins.append(FRONTEND_URL.rstrip("/"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# OPENAI CLIENT
# ============================================================

openai_client = None

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# REQUEST MODELS
# ============================================================

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
    # Origin
    origin_text: Optional[str] = None
    origin_zip: Optional[str] = None

    # Destination
    destination_city: Optional[str] = None
    destination_country: Optional[str] = None
    destination_region: Optional[str] = None

    destination_preferences: list[str] = Field(default_factory=list)

    # Dates
    date_start: Optional[str] = None
    date_end: Optional[str] = None

    # Budget
    budget_usd: Optional[float] = None
    budget_is_per_person: Optional[bool] = None

    # Travelers
    passengers: Optional[int] = None
    cabin: Optional[str] = None

    # Flight preferences
    max_stops: Optional[int] = None
    checked_bags: Optional[int] = None

    # Airport preferences
    airport_radius_miles: Optional[int] = None


class AIChatRequest(BaseModel):
    message: str

    state: TripState = Field(
        default_factory=TripState
    )


class AIChatResponse(BaseModel):
    message: str
    state: TripState
    ready_to_search: bool

    missing_fields: list[str] = Field(
        default_factory=list
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "SkyOpt IQ backend is running",
        "version": "2.0.0",
        "services": {
            "flight_search": True,
            "ai_chat": bool(OPENAI_API_KEY),
        },
    }


@app.get("/api/v1/health")
def health():
    return {
        "status": "ok",
        "serpapi_configured": bool(SERPAPI_KEY),
        "openai_configured": bool(OPENAI_API_KEY),
    }


# ============================================================
# HELPERS
# ============================================================

def get_travel_class(cabin: str) -> int:
    """
    Google Flights travel_class values:

    1 = Economy
    2 = Premium Economy
    3 = Business
    4 = First
    """

    cabin = (cabin or "economy").lower()

    mapping = {
        "economy": 1,
        "premium economy": 2,
        "premium_economy": 2,
        "premium": 2,
        "business": 3,
        "first": 4,
        "first class": 4,
    }

    return mapping.get(cabin, 1)


def normalize_flight(entry: dict) -> Optional[dict]:
    """
    Converts a SerpAPI Google Flights result into
    the format expected by the SkyOpt frontend.
    """

    flights = entry.get("flights", [])

    if not flights:
        return None

    first_leg = flights[0]
    last_leg = flights[-1]

    airline = first_leg.get("airline") or "Unknown"

    return {
        "source": "serp",

        "price": entry.get("price", 0),

        "airline": airline,

        "logo": first_leg.get("airline_logo"),

        "stops": max(len(flights) - 1, 0),

        "duration": entry.get("total_duration", 0),

        "dep_time": (
            first_leg
            .get("departure_airport", {})
            .get("time")
        ),

        "arr_time": (
            last_leg
            .get("arrival_airport", {})
            .get("time")
        ),

        "flight_code": first_leg.get(
            "flight_number"
        ),

        "legs": flights,

        "layovers": entry.get(
            "layovers",
            []
        ),

        "extensions": entry.get(
            "extensions",
            []
        ),

        "fare_name": entry.get(
            "fare_name"
        ),

        "carbon_emissions": entry.get(
            "carbon_emissions"
        ),
    }


# ============================================================
# FLIGHT SEARCH
# ============================================================

@app.post("/api/v1/flights/search")
def search_flights(payload: FlightSearchRequest):

    if not SERPAPI_KEY:
        raise HTTPException(
            status_code=500,
            detail="SERPAPI_KEY is missing",
        )

    trip_type = payload.trip_type.lower()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    if trip_type not in {
        "oneway",
        "roundtrip",
        "multi",
    }:
        raise HTTPException(
            status_code=400,
            detail="trip_type must be oneway, roundtrip, or multi",
        )

    if trip_type == "roundtrip" and not payload.return_date:
        raise HTTPException(
            status_code=400,
            detail="return_date is required for roundtrip",
        )

    # --------------------------------------------------------
    # Google Flights parameters
    # --------------------------------------------------------

    params = {
        "engine": "google_flights",

        "departure_id": payload.origin.upper(),
        "arrival_id": payload.destination.upper(),

        "outbound_date": payload.depart_date,

        "currency": "USD",
        "hl": "en",

        "adults": max(payload.passengers, 1),

        "travel_class": get_travel_class(
            payload.cabin
        ),
    }

    # --------------------------------------------------------
    # Trip type
    # --------------------------------------------------------

    if trip_type == "roundtrip":

        # Google Flights:
        # type 1 = round trip

        params["type"] = 1
        params["return_date"] = payload.return_date

    else:

        # Google Flights:
        # type 2 = one way

        params["type"] = 2

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    try:

        client = serpapi.Client(
            api_key=SERPAPI_KEY
        )

        results = client.search(params)

        if "error" in results:
            raise HTTPException(
                status_code=502,
                detail=results["error"],
            )

        best = results.get(
            "best_flights",
            []
        )

        others = results.get(
            "other_flights",
            []
        )

        combined = best + others

        normalized = []

        for entry in combined:

            flight = normalize_flight(entry)

            if flight:
                normalized.append(flight)

        return {
            "origin": payload.origin.upper(),
            "destination": payload.destination.upper(),
            "trip_type": trip_type,
            "depart_date": payload.depart_date,
            "return_date": payload.return_date,
            "flights": normalized,
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Flight search error: {str(e)}",
        )


# ============================================================
# SKYOPT AI SYSTEM PROMPT
# ============================================================

SKYOPT_INSTRUCTIONS = """
You are SkyOpt IQ, an intelligent travel-search assistant.

Your job is to understand what trip a traveler wants and
progressively construct a structured trip state.

You are NOT responsible for inventing flights, prices,
airports, schedules, or availability.

Actual flight data comes from SkyOpt's travel APIs.


GENERAL BEHAVIOR

1. Speak naturally and concisely.

2. Ask only ONE important follow-up question at a time.

3. Never ask for information already contained in the trip
   state unless clarification is genuinely necessary.

4. Update the state whenever the traveler gives new
   information.

5. Preserve useful information from the existing state.

6. Do not reset fields unless the traveler explicitly changes
   them.


ORIGIN RULES

Travelers are NOT expected to know airport codes.

Never require the traveler to provide an airport code.

Accept origins such as:

"Keene, NH"
"New York"
"03431"
"Boston"
"USA"
"I live near Chicago"

If the traveler gives a sufficiently specific city or town,
store it in origin_text.

Example:

"I live in Keene, NH"

origin_text = "Keene, NH"

Do NOT ask for ZIP code if the city/town is already specific
enough for geocoding.

If the traveler only provides something broad such as:

"USA"
"California"
"the northeast"

ask for their ZIP code or city/town.

Another SkyOpt service will later convert the origin into
coordinates and nearby commercial airports.


DESTINATION RULES

The destination does NOT have to be a specific airport or
city.

Understand destination discovery requests such as:

"somewhere warm in Europe"
"a cheap beach destination"
"somewhere tropical"
"somewhere in Asia"
"somewhere warm for Christmas"
"a romantic city in Europe"
"a beach destination in the Caribbean"

Use:

destination_city
destination_country
destination_region
destination_preferences

appropriately.

Example:

"I want somewhere warm in Europe"

destination_region = "Europe"

destination_preferences = ["warm"]

Do NOT invent the final destination.

Another SkyOpt service will generate and evaluate candidate
destinations.


DATE RULES

Understand explicit dates whenever possible.

If the traveler gives:

"December 23 to December 31"

store the corresponding dates.

Use YYYY-MM-DD.

Use the most contextually reasonable upcoming year if the
year is omitted.

If the traveler only says:

"Christmas"
"spring break"
"next summer"

and exact dates are necessary, ask about their flexible date
range.

Do not pretend an ambiguous holiday phrase represents exact
travel dates.


BUDGET RULES

Understand phrases such as:

"$1300"
"under $1,300"
"my budget is 2k"
"I don't want to spend more than 900 dollars"

Store the numeric value in budget_usd.

If it matters and is unclear whether the budget is per
traveler or total, ask later.

Do not repeatedly ask about budget.


PASSENGERS

If the traveler says:

"my wife and I"

passengers = 2

If they say:

"me and my two kids"

passengers = 3

If passenger count is unknown, it does not always need to
block destination discovery.


FLIGHT PREFERENCES

Understand preferences such as:

"nonstop only"

max_stops = 0

"maximum one stop"

max_stops = 1

"I have two checked bags"

checked_bags = 2


AIRPORT RADIUS

If the traveler specifies how far they are willing to travel
to an airport, store airport_radius_miles.

Examples:

"within 100 miles"

airport_radius_miles = 100

"I'll drive up to 3 hours"

Do not convert driving hours into miles yourself.
Ask for clarification later if necessary.


SEARCH READINESS

ready_to_search should be true when SkyOpt has enough
information to begin destination discovery or flight search.

Normally this requires:

1. A usable origin location.

AND

2. Either:
   - destination_city
   - destination_country
   - destination_region

AND

3. usable travel dates.

Budget is useful but is not always mandatory.

Airport radius is useful but is not always mandatory.

Passenger count is useful but does not always need to block
initial discovery.


CRITICAL SAFETY / ACCURACY RULE

Never claim that a flight exists.

Never invent airfare.

Never invent flight schedules.

Never say that you "found" a flight unless flight-search
results were actually provided to you.

Your role in this endpoint is to understand the traveler's
request and determine what information SkyOpt needs next.
"""


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

TRIP_SCHEMA = {
    "type": "object",

    "properties": {

        "message": {
            "type": "string"
        },

        "state": {
            "type": "object",

            "properties": {

                "origin_text": {
                    "type": ["string", "null"]
                },

                "origin_zip": {
                    "type": ["string", "null"]
                },

                "destination_city": {
                    "type": ["string", "null"]
                },

                "destination_country": {
                    "type": ["string", "null"]
                },

                "destination_region": {
                    "type": ["string", "null"]
                },

                "destination_preferences": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },

                "date_start": {
                    "type": ["string", "null"]
                },

                "date_end": {
                    "type": ["string", "null"]
                },

                "budget_usd": {
                    "type": ["number", "null"]
                },

                "budget_is_per_person": {
                    "type": ["boolean", "null"]
                },

                "passengers": {
                    "type": ["integer", "null"]
                },

                "cabin": {
                    "type": ["string", "null"]
                },

                "max_stops": {
                    "type": ["integer", "null"]
                },

                "checked_bags": {
                    "type": ["integer", "null"]
                },

                "airport_radius_miles": {
                    "type": ["integer", "null"]
                },
            },

            "required": [
                "origin_text",
                "origin_zip",
                "destination_city",
                "destination_country",
                "destination_region",
                "destination_preferences",
                "date_start",
                "date_end",
                "budget_usd",
                "budget_is_per_person",
                "passengers",
                "cabin",
                "max_stops",
                "checked_bags",
                "airport_radius_miles",
            ],

            "additionalProperties": False,
        },

        "ready_to_search": {
            "type": "boolean"
        },

        "missing_fields": {
            "type": "array",
            "items": {
                "type": "string"
            }
        },
    },

    "required": [
        "message",
        "state",
        "ready_to_search",
        "missing_fields",
    ],

    "additionalProperties": False,
}


# ============================================================
# AI CHAT
# ============================================================

@app.post(
    "/api/v1/ai/chat",
    response_model=AIChatResponse,
)
def ai_chat(payload: AIChatRequest):

    if not OPENAI_API_KEY or openai_client is None:

        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is missing",
        )

    message = payload.message.strip()

    if not message:

        raise HTTPException(
            status_code=400,
            detail="message cannot be empty",
        )

    current_state = payload.state.model_dump_json(
        indent=2
    )

    user_input = f"""
CURRENT TRIP STATE:

{current_state}


NEW TRAVELER MESSAGE:

{message}


Update the trip state using the new message.

Keep information from CURRENT TRIP STATE unless the traveler
explicitly changes it.

Determine whether enough information exists to begin search.

If information is missing, ask ONE useful follow-up question.
"""

    try:

        response = openai_client.responses.create(

            model=OPENAI_MODEL,

            instructions=SKYOPT_INSTRUCTIONS,

            input=user_input,

            text={
                "format": {
                    "type": "json_schema",
                    "name": "skyopt_trip_state",
                    "strict": True,
                    "schema": TRIP_SCHEMA,
                }
            },
        )

        if not response.output_text:

            raise HTTPException(
                status_code=502,
                detail="OpenAI returned an empty response",
            )

        result = json.loads(
            response.output_text
        )

        return AIChatResponse(
            **result
        )

    except HTTPException:
        raise

    except json.JSONDecodeError as e:

        raise HTTPException(
            status_code=502,
            detail=f"Invalid AI JSON response: {str(e)}",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"SkyOpt AI error: {str(e)}",
        )
