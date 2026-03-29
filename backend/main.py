import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from serpapi import GoogleSearch

load_dotenv()

app = FastAPI(title="SkyOpt-IQ Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        # add your real Netlify URL here after deploy
        # "https://your-site-name.netlify.app",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY")


class FlightSearchRequest(BaseModel):
    origin: str
    destination: str
    depart_date: str
    cabin: str | None = "economy"
    passengers: int | None = 1
    weight_price: float | None = 0.5
    weight_stops: float | None = 0.3
    weight_duration: float | None = 0.2


@app.get("/")
def root():
    return {"message": "SkyOpt-IQ backend is running"}


@app.post("/api/v1/flights/search")
def search_flights(payload: FlightSearchRequest):
    if not SERPAPI_KEY:
        raise HTTPException(status_code=500, detail="SERPAPI_KEY is missing")

    params = {
        "engine": "google_flights",
        "departure_id": payload.origin,
        "arrival_id": payload.destination,
        "outbound_date": payload.depart_date,
        "currency": "USD",
        "hl": "en",
        "type": 2,
        "api_key": SERPAPI_KEY,
    }

    try:
        results = GoogleSearch(params).get_dict()

        if "error" in results:
            raise HTTPException(status_code=502, detail=results["error"])

        best = results.get("best_flights", [])
        others = results.get("other_flights", [])
        combined = best + others

        normalized = []

        for entry in combined:
            flights = entry.get("flights", [])
            if not flights:
                continue

            first_leg = flights[0]
            last_leg = flights[-1]

            normalized.append({
                "source": "serp",
                "price": entry.get("price", 0),
                "airline": first_leg.get("airline"),
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
            })

        return {"flights": normalized}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))