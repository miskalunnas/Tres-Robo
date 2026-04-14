"""Fetch upcoming TRES events from the Luma public calendar.

Discovers the calendar API ID from lu.ma/tres on first call and caches it
in-process. Results are cached for CACHE_TTL_SECONDS to avoid hammering
the API on repeated questions within a session.

Environment variables:
    LUMA_API_KEY       — Luma API key (required, from lu.ma → Settings → API)
    LUMA_CALENDAR_ID   — override auto-discovery (format: cal-xxxxxxxxx)
    LUMA_CALENDAR_SLUG — Luma URL slug (default: tres)
"""

import os
import re
import sys
import time as _time
from datetime import datetime, timezone, timedelta

import requests

_SLUG = os.environ.get("LUMA_CALENDAR_SLUG", "tres")
_LUMA_API = "https://api.lu.ma/public/v1"
_LUMA_PAGE = f"https://lu.ma/{_SLUG}"
CACHE_TTL_SECONDS = 900  # 15 minutes


def _api_headers() -> dict:
    """Build request headers, including the API key if set."""
    headers = {"accept": "application/json"}
    key = os.environ.get("LUMA_API_KEY", "").strip()
    if key:
        headers["x-luma-api-key"] = key
    return headers

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Helsinki")
except ImportError:
    _TZ = timezone(timedelta(hours=3))  # UTC+3 rough fallback

# ── In-process cache ──────────────────────────────────────────────────────────
_calendar_id: str | None = os.environ.get("LUMA_CALENDAR_ID", "").strip() or None
_cache_text: str | None = None
_cache_expires: float = 0.0


def _discover_calendar_id() -> str | None:
    """Extract cal-xxx from the lu.ma calendar page HTML."""
    try:
        r = requests.get(_LUMA_PAGE, timeout=8, headers={"User-Agent": "Tres-Robo/1.0", **_api_headers()})
        r.raise_for_status()
        match = re.search(r'"(cal-[A-Za-z0-9]+)"', r.text)
        if match:
            return match.group(1)
        # Fallback: look for calendar_api_id in JSON blobs
        match = re.search(r'calendar_api_id["\s:]+(["\'])(cal-[A-Za-z0-9]+)\1', r.text)
        if match:
            return match.group(2)
    except Exception as exc:
        print(f"[Events] calendar ID discovery failed: {exc}", file=sys.stderr)
    return None


def _parse_dt(iso: str) -> datetime | None:
    """Parse ISO-8601 string to an aware datetime."""
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.astimezone(_TZ)
    except Exception:
        return None


def _format_event(event: dict, include_id: bool = False) -> str:
    """Format a single event dict to a compact one-line string.

    When include_id=True, appends [id:evt-xxx] so the model can pass the
    exact ID to get_event_details without any ambiguity.
    """
    name = event.get("name") or "Untitled"
    start_iso = event.get("start_at") or ""
    dt = _parse_dt(start_iso)

    date_str = dt.strftime("%-d %b %Y %H:%M") if dt else "date TBD"

    geo = event.get("geo_address_info") or {}
    location = (
        (event.get("geo_address_json") or {}).get("name")
        or geo.get("address")
        or geo.get("city")
        or ""
    ).strip()

    line = f"{name} — {date_str}"
    if location:
        line += f" @ {location}"
    if include_id:
        event_id = event.get("api_id") or ""
        if event_id:
            line += f" [id:{event_id}]"
    return line


def get_upcoming_events(limit: int = 6) -> str:
    """Return a compact list of upcoming TRES events with embedded IDs.

    Each line includes [id:evt-xxx] so the model can call get_event_details
    with the exact ID — no name-matching ambiguity.
    Uses a 15-minute in-process cache.
    """
    global _calendar_id, _cache_text, _cache_expires

    now = _time.monotonic()
    if _cache_text and now < _cache_expires:
        return _cache_text

    if not _calendar_id:
        _calendar_id = _discover_calendar_id()
    if not _calendar_id:
        return "Luma-kalenterin tunnistetta ei löytynyt. Tarkista LUMA_CALENDAR_ID."

    try:
        r = requests.get(
            f"{_LUMA_API}/calendar/list-events",
            params={"calendar_api_id": _calendar_id},
            timeout=8,
            headers=_api_headers(),
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"[Events] Luma API error: {exc}", file=sys.stderr)
        return "Tapahtumatietojen haku epäonnistui. Yritä hetken kuluttua uudelleen."

    entries = data.get("entries") or []
    now_tz = datetime.now(timezone.utc).astimezone(_TZ)

    upcoming = []
    for entry in entries:
        event = entry.get("event") or {}
        dt = _parse_dt(event.get("start_at") or "")
        if dt and dt > now_tz:
            upcoming.append((dt, event))

    upcoming.sort(key=lambda x: x[0])
    upcoming = upcoming[:limit]

    if not upcoming:
        result = "Ei tulevia TRES-tapahtumia Luma-kalenterissa."
    else:
        lines = [f"Tulevat TRES-tapahtumat ({len(upcoming)} kpl):"]
        for _, event in upcoming:
            lines.append(f"• {_format_event(event, include_id=True)}")
        result = "\n".join(lines)

    _cache_text = result
    _cache_expires = now + CACHE_TTL_SECONDS
    return result


def get_event_details(event_id: str) -> str:
    """Fetch full details for a single event by its Luma API ID (evt-xxx).

    Returns a compact summary including description, location, and URL.
    """
    if not event_id or not event_id.startswith("evt-"):
        return "Virheellinen tapahtuma-ID. Kutsu ensin get_events saadaksesi oikeat ID:t."

    try:
        r = requests.get(
            f"{_LUMA_API}/event/get",
            params={"api_id": event_id},
            timeout=8,
            headers=_api_headers(),
        )
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        print(f"[Events] event detail fetch failed ({event_id}): {exc}", file=sys.stderr)
        return "Tapahtuman tietojen haku epäonnistui."

    event = data.get("event") or data  # API may return event at top level

    name = event.get("name") or "Untitled"
    dt = _parse_dt(event.get("start_at") or "")
    date_str = dt.strftime("%-d %b %Y %H:%M") if dt else "date TBD"

    geo = event.get("geo_address_info") or {}
    location = (
        (event.get("geo_address_json") or {}).get("name")
        or geo.get("full_address")
        or geo.get("address")
        or geo.get("city")
        or ""
    ).strip()

    description = (event.get("description") or "").strip()
    url = event.get("url") or f"https://lu.ma/{event.get('slug', '')}"

    parts = [f"{name} — {date_str}"]
    if location:
        parts.append(f"Paikka: {location}")
    if description:
        # Trim to ~400 chars — enough context without overwhelming the model
        trimmed = description[:400]
        if len(description) > 400:
            trimmed += "…"
        parts.append(f"Kuvaus: {trimmed}")
    if url:
        parts.append(f"Lisätietoja: {url}")

    return "\n".join(parts)
