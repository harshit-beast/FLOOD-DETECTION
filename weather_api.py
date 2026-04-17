"""Helpers for fetching near real-time weather data from Open-Meteo."""

from __future__ import annotations

import json
import time
from typing import Any, Dict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_HEADERS = {"User-Agent": "FloodDetectionApp/1.0 (streamlit)"}


class WeatherAPIRateLimitError(RuntimeError):
    """Raised when the weather API returns HTTP 429 after retries."""

    def __init__(self, wait_seconds: int | None = None):
        self.wait_seconds = wait_seconds
        if wait_seconds:
            super().__init__(f"Rate limited by weather API. Retry after about {wait_seconds} seconds.")
        else:
            super().__init__("Rate limited by weather API. Please retry in a short while.")


def _parse_retry_after(header_value: str | None, fallback_seconds: float) -> int:
    """Parse Retry-After header value into a whole-second wait duration."""
    if not header_value:
        return max(1, int(round(fallback_seconds)))
    try:
        return max(1, int(float(header_value)))
    except ValueError:
        return max(1, int(round(fallback_seconds)))


def _get_json(
    url: str,
    params: Dict[str, Any],
    timeout: int = 12,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> Dict[str, Any]:
    """Run a GET request and decode a JSON response."""
    query = urlencode(params, doseq=True)
    endpoint = f"{url}?{query}"
    request = Request(endpoint, headers=DEFAULT_HEADERS)

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - trusted API endpoint.
                payload = response.read()
            return json.loads(payload.decode("utf-8"))
        except HTTPError as exc:
            should_retry = attempt < retries
            wait_guess = backoff_seconds * (2**attempt)

            if exc.code == 429:
                wait_seconds = _parse_retry_after(exc.headers.get("Retry-After"), wait_guess)
                if should_retry:
                    time.sleep(wait_seconds)
                    continue
                raise WeatherAPIRateLimitError(wait_seconds=wait_seconds) from exc

            if 500 <= exc.code <= 599 and should_retry:
                time.sleep(wait_guess)
                continue
            raise
        except URLError:
            if attempt < retries:
                time.sleep(backoff_seconds * (2**attempt))
                continue
            raise

    raise RuntimeError("Unexpected API retry loop exit.")


def geocode_location(location_name: str) -> Dict[str, Any]:
    """Resolve free-text location to coordinates using Open-Meteo geocoding."""
    response = _get_json(
        GEOCODE_URL,
        {
            "name": location_name,
            "count": 1,
            "language": "en",
            "format": "json",
        },
    )
    results = response.get("results") or []
    if not results:
        raise ValueError(f"Could not resolve location: {location_name}")
    return results[0]


def fetch_live_weather(latitude: float, longitude: float) -> Dict[str, float]:
    """Fetch current weather plus hourly context used for flood-risk features."""
    response = _get_json(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure",
            "hourly": "precipitation,soil_moisture_0_to_1cm",
            "forecast_days": 1,
        },
    )

    current = response.get("current") or {}
    hourly = response.get("hourly") or {}
    hourly_rain = hourly.get("precipitation") or []
    hourly_soil = hourly.get("soil_moisture_0_to_1cm") or []

    rainfall_24h_mm = float(sum(hourly_rain[:24])) if hourly_rain else float(current.get("precipitation", 0.0))
    soil_moisture_pct = float(hourly_soil[0] * 100.0) if hourly_soil else 0.0

    return {
        "temperature_c": float(current.get("temperature_2m", 0.0)),
        "humidity_pct": float(current.get("relative_humidity_2m", 0.0)),
        "precipitation_mm": float(current.get("precipitation", 0.0)),
        "rainfall_24h_mm": rainfall_24h_mm,
        "soil_moisture_pct": soil_moisture_pct,
        "wind_speed_kmh": float(current.get("wind_speed_10m", 0.0)),
        "surface_pressure_hpa": float(current.get("surface_pressure", 0.0)),
    }


def fetch_weather_for_location(location_name: str) -> Dict[str, Any]:
    """Resolve a location and return location metadata plus weather readings."""
    geo = geocode_location(location_name)
    weather = fetch_live_weather(float(geo["latitude"]), float(geo["longitude"]))

    location_label = ", ".join(
        part
        for part in [
            geo.get("name"),
            geo.get("admin1"),
            geo.get("country"),
        ]
        if part
    )

    return {
        "location": location_label or location_name,
        "latitude": float(geo["latitude"]),
        "longitude": float(geo["longitude"]),
        **weather,
    }


def fetch_hourly_weather_history(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Fetch hourly historical weather used to train a location-specific flood model."""
    response = _get_json(
        ARCHIVE_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": "auto",
            "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,surface_pressure",
        },
    )

    hourly = response.get("hourly") or {}
    if not hourly or not hourly.get("time"):
        raise ValueError("No historical weather records were returned for this location/date range.")

    return hourly
