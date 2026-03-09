"""Menu checker tool for Hervanta campus lunch menus via Unisafka."""

import sys
from datetime import date

import requests

UNISAFKA_BASE = "https://unisafka.fi/static/json"
CAMPUS = "tty"

# Erikoisruokavalioiden lyhenteet → selitetty muoto (botti ja TTS lukevat sujuvasti)
DIET_CODES: dict[str, str] = {
    "g": "gluteeniton",
    "l": "laktoositon",
    "vl": "vähälaktoosinen",
    "vg": "vegaani",
    "v": "vege",
    "m": "maidoton",
    "s": "soijaton",
    "sml": "sianliha",
    "k": "kananmuna",
}

# Päivämäärän puhemuoto (ääntäminen) — kuukaudet genetiivissä
MONTHS_FI = (
    "tammikuuta", "helmikuuta", "maaliskuuta", "huhtikuuta", "toukokuuta", "kesäkuuta",
    "heinäkuuta", "elokuuta", "syyskuuta", "lokakuuta", "marraskuuta", "joulukuuta",
)
WEEKDAYS_FI = ("maanantai", "tiistai", "keskiviikko", "torstai", "perjantai", "lauantai", "sunnuntai")


def _spoken_date(d: date) -> str:
    """Päivämäärä luettavassa muodossa: maanantai 9. maaliskuuta 2025."""
    wd = WEEKDAYS_FI[d.weekday()]
    mo = MONTHS_FI[d.month - 1]
    return f"{wd} {d.day}. {mo} {d.year}"

RESTAURANT_KEYS: dict[str, tuple[str, ...]] = {
    "reaktori": ("res_reaktori", "res_reaktori_iltaruoka"),
    "newton": ("res_newton",),
    "konehuone": ("res_konehuone", "res_newton_soos", "res_newton_fusion"),
    "hertsi": ("res_hertsi",),
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
    "fazer": "reaktori",
    "break cafe": "reaktori",
    "cafe konehuone": "konehuone",
    "café konehuone": "konehuone",
}

RESTAURANT_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "reaktori": ("reaktori",),
    "newton": ("newton",),
    "konehuone": ("konehuone",),
    "hertsi": ("hertsi",),
}

_DAYS_FI = ("ma", "ti", "ke", "to", "pe", "la", "su")


def _expand_diets(raw: str) -> str:
    """Expand diet codes to readable Finnish (e.g. 'G, L' -> 'gluteeniton, laktoositon')."""
    if not raw or not raw.strip():
        return ""
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    expanded = [DIET_CODES.get(p, p.upper()) for p in parts]
    return ", ".join(expanded)


def _resolve_name(name: str) -> str | None:
    """Map a user-given name to a canonical restaurant key, or None."""
    key = name.strip().lower()
    if key in RESTAURANT_KEYS:
        return key
    return ALIASES.get(key)


def _week_number() -> int:
    """Return the ISO week number used by Unisafka's JSON files."""
    return date.today().isocalendar().week


def _fetch_json(url: str) -> dict:
    """Fetch JSON from *url* and raise on HTTP errors."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _fetch_day_data() -> dict:
    """Fetch today's Unisafka JSON payload."""
    today = date.today()
    year = today.year
    week = _week_number()
    day = _DAYS_FI[today.weekday()]

    v_url = f"{UNISAFKA_BASE}/{year}/{week}/v.json"
    version_data = _fetch_json(v_url)
    version = str(version_data.get("v", "")).strip()
    if not version:
        raise ValueError(f"Missing version in {v_url}")

    menu_url = f"{UNISAFKA_BASE}/{year}/{week}/{version}/{day}.json"
    return _fetch_json(menu_url)


def _get_restaurants(data: dict) -> dict[str, dict]:
    return data.get(f"restaurants_{CAMPUS}", {})


def _find_restaurant_entries(data: dict, key: str) -> list[dict]:
    """Return matching restaurant entries for a canonical restaurant name."""
    restaurants = _get_restaurants(data)
    matches: list[dict] = []
    seen: set[str] = set()

    for res_key in RESTAURANT_KEYS[key]:
        rest = restaurants.get(res_key)
        if rest:
            matches.append(rest)
            seen.add(res_key)

    if matches:
        return matches

    hints = RESTAURANT_NAME_HINTS[key]
    for res_key, rest in restaurants.items():
        if res_key in seen:
            continue
        name = (rest.get("restaurant") or "").lower()
        if any(hint in name for hint in hints):
            matches.append(rest)

    return matches


def _format_meal_lines(rest: dict, *, include_restaurant_name: bool) -> list[str]:
    """Extract formatted lines for one Unisafka restaurant object."""
    restaurant_name = (rest.get("restaurant") or "Unknown restaurant").strip()
    meal_indent = "    " if include_restaurant_name else "  "
    item_indent = "      " if include_restaurant_name else "    "

    if not rest.get("open_today", False):
        if include_restaurant_name:
            return [f"  {restaurant_name}: Closed today."]
        return ["  Closed today."]

    lines: list[str] = []
    if include_restaurant_name:
        lines.append(f"  {restaurant_name}:")

    for meal in rest.get("meals", []):
        category = (meal.get("kok") or "").strip()
        items = meal.get("mo", [])
        if category:
            lines.append(f"{meal_indent}{category}:")
        for item in items:
            name = (item.get("mpn") or "").strip()
            diets_raw = (item.get("mpd") or "").strip()
            if not name:
                continue
            entry = f"{item_indent}{name}"
            if diets_raw:
                diets_readable = _expand_diets(diets_raw)
                entry += f" — {diets_readable}"
            lines.append(entry)

    if include_restaurant_name and len(lines) == 1:
        lines.append(f"{meal_indent}No menu available.")

    return lines


def _parse_restaurant(data: dict, key: str) -> list[str]:
    """Extract formatted menu lines for a canonical restaurant key."""
    matches = _find_restaurant_entries(data, key)
    if not matches:
        return []

    include_restaurant_name = len(matches) > 1
    lines: list[str] = []
    for index, rest in enumerate(matches):
        block = _format_meal_lines(rest, include_restaurant_name=include_restaurant_name)
        if not block:
            continue
        if lines and include_restaurant_name and index > 0:
            lines.append("")
        lines.extend(block)
    return lines


def _menu_fetch_error(display: str | None = None) -> str:
    if display:
        return f"Failed to fetch menu for {display}. Check unisafka.fi."
    return "Failed to fetch menus. Check unisafka.fi."


def get_menu(restaurant: str) -> str:
    """Fetch today's lunch menu for *restaurant* and return readable text."""
    key = _resolve_name(restaurant)
    if key is None:
        available = ", ".join(RESTAURANT_DISPLAY.values())
        return f"Unknown restaurant: '{restaurant}'. Options: {available}"

    display = RESTAURANT_DISPLAY[key]

    try:
        data = _fetch_day_data()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        print(f"[Menu] HTTP {status} while fetching {display}: {exc}", file=sys.stderr)
        return _menu_fetch_error(display)
    except requests.RequestException as exc:
        print(f"[Menu] Request error while fetching {display}: {exc}", file=sys.stderr)
        return _menu_fetch_error(display)
    except Exception as exc:
        print(f"[Menu] Unexpected error while fetching {display}: {exc}", file=sys.stderr)
        return _menu_fetch_error(display)

    lines = _parse_restaurant(data, key)
    if not lines:
        return f"No menu found for {display} today."

    today = date.today()
    spoken = _spoken_date(today)
    header = f"Päivämäärä: {spoken}.\n{display} — {today.strftime('%d.%m.%Y')}:"
    return header + "\n" + "\n".join(lines)


def get_all_menus() -> str:
    """Fetch today's lunch menus for all supported restaurants."""
    try:
        data = _fetch_day_data()
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        print(f"[Menu] HTTP {status} while fetching all menus: {exc}", file=sys.stderr)
        return _menu_fetch_error()
    except requests.RequestException as exc:
        print(f"[Menu] Request error while fetching all menus: {exc}", file=sys.stderr)
        return _menu_fetch_error()
    except Exception as exc:
        print(f"[Menu] Unexpected error while fetching all menus: {exc}", file=sys.stderr)
        return _menu_fetch_error()

    parts: list[str] = []
    today = date.today()
    spoken = _spoken_date(today)
    header = f"Päivämäärä: {spoken}.\nLunch menus — {today.strftime('%d.%m.%Y')}"
    for key in RESTAURANT_DISPLAY:
        display = RESTAURANT_DISPLAY[key]
        lines = _parse_restaurant(data, key)
        if lines:
            parts.append(f"{display}:\n" + "\n".join(lines))
        else:
            parts.append(f"{display}: No menu available.")

    return header + "\n\n" + "\n\n".join(parts)


def list_restaurants() -> list[str]:
    """Return the canonical names of all supported restaurants."""
    return list(RESTAURANT_DISPLAY.values())
