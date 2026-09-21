import os
import json
from typing import Optional, Any
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
            "message": "SkyOpt IQ is connected to Duffel",
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
        return {
            "status": "confirmed",
            "test_mode": True,
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

