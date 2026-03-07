"""Menu checker tool — fetches today's lunch menu from Tampere campus
restaurants via the Unisafka.fi static JSON API.

Supported restaurants: Reaktori, Newton, Konehuone, Hertsi
"""

import math
import sys
from datetime import date

import requests

UNISAFKA_BASE = "https://unisafka.fi/static/json"
CAMPUS = "tty"

RESTAURANT_KEYS: dict[str, str] = {
    "reaktori": "res_reaktori",
    "newton": "res_newton",
    "konehuone": "res_konehuone",
    "hertsi": "res_hertsi",
}

RESTAURANT_DISPLAY: dict[str, str] = {
    "reaktori": "Reaktori",
    "newton": "Newton",
    "konehuone": "Konehuone",
    "hertsi": "Hertsi",
}

ALIASES: dict[str, str] = {
    "food & co": "reaktori",
    "food and co": "reaktori",
    "foodco": "reaktori",
    "cafe konehuone": "konehuone",
    "café konehuone": "konehuone",
}


def _resolve_name(name: str) -> str | None:
    """Map a user-given name to a canonical restaurant key, or None."""
    key = name.strip().lower()
    if key in RESTAURANT_KEYS:
        return key
    return ALIASES.get(key)


def _week_number() -> int:
    """Replicate the Unisafka week-number algorithm (days since Jan 1 / 7, ceil)."""
    today = date.today()
    days = (today - date(today.year, 1, 1)).days
    week = math.ceil(days / 7)
    if today.weekday() == 6:
        week += 1
    return week


_DAYS_FI = ("ma", "ti", "ke", "to", "pe", "la", "su")


def _fetch_day_data() -> dict:
    """Fetch the full Unisafka JSON for today."""
    year = date.today().year
    week = _week_number()

    v_url = f"{UNISAFKA_BASE}/{year}/{week}/v.json"
    r = requests.get(v_url, timeout=10)
    r.raise_for_status()
    version = r.json()["v"]

    day = _DAYS_FI[date.today().weekday()]
    menu_url = f"{UNISAFKA_BASE}/{year}/{week}/{version}/{day}.json"
    r = requests.get(menu_url, timeout=10)
    r.raise_for_status()
    return r.json()


def _parse_restaurant(data: dict, res_key: str) -> list[str]:
    """Extract formatted menu lines for one restaurant from Unisafka JSON."""
    restaurants = data.get(f"restaurants_{CAMPUS}", {})
    rest = restaurants.get(res_key)
    if not rest:
        return []
    if not rest.get("open_today", False):
        return ["  Closed today."]

    lines: list[str] = []
    for meal in rest.get("meals", []):
        category = meal.get("kok", "")
        items = meal.get("mo", [])
        if category:
            lines.append(f"  {category}:")
        for item in items:
            name = item.get("mpn", "").strip()
            diets = item.get("mpd", "")
            if name:
                entry = f"    {name}"
                if diets:
                    entry += f" ({diets})"
                lines.append(entry)
    return lines


def get_menu(restaurant: str) -> str:
    """Fetch today's lunch menu for *restaurant* and return readable text."""
    key = _resolve_name(restaurant)
    if key is None:
        available = ", ".join(RESTAURANT_DISPLAY.values())
        return f"Unknown restaurant: '{restaurant}'. Options: {available}"

    res_key = RESTAURANT_KEYS[key]
    display = RESTAURANT_DISPLAY[key]

    try:
        data = _fetch_day_data()
    except requests.RequestException as exc:
        print(f"[Menu] HTTP error: {exc}", file=sys.stderr)
        return f"Failed to fetch menu for {display}."
    except Exception as exc:
        print(f"[Menu] Unexpected error: {exc}", file=sys.stderr)
        return f"Failed to fetch menu for {display}."

    lines = _parse_restaurant(data, res_key)
    if not lines:
        return f"No menu found for {display} today."

    header = f"{display} — {date.today().strftime('%d.%m.%Y')}:"
    return header + "\n" + "\n".join(lines)


def get_all_menus() -> str:
    """Fetch today's lunch menus for all restaurants."""
    try:
        data = _fetch_day_data()
    except requests.RequestException as exc:
        print(f"[Menu] HTTP error: {exc}", file=sys.stderr)
        return "Failed to fetch menus."
    except Exception as exc:
        print(f"[Menu] Unexpected error: {exc}", file=sys.stderr)
        return "Failed to fetch menus."

    parts: list[str] = []
    for key, res_key in RESTAURANT_KEYS.items():
        display = RESTAURANT_DISPLAY[key]
        lines = _parse_restaurant(data, res_key)
        if lines:
            parts.append(f"{display}:\n" + "\n".join(lines))
        else:
            parts.append(f"{display}: No menu available.")

    header = f"Lunch menus — {date.today().strftime('%d.%m.%Y')}"
    return header + "\n\n" + "\n\n".join(parts)


def list_restaurants() -> list[str]:
    """Return the canonical names of all supported restaurants."""
    return list(RESTAURANT_DISPLAY.values())
