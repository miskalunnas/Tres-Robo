"""Menu checker tool — fetches today's lunch menu from Tampere campus restaurants.

Supported restaurants:
  - Reaktori  (Compass Group / Food & Co)
  - Newton    (Juvenes / Jamix)
  - Konehuone (Juvenes / Jamix)
  - Hertsi    (Sodexo)
"""

import sys
from datetime import date

import requests

# ── Restaurant configuration ─────────────────────────────────────

RESTAURANTS: dict[str, dict] = {
    "reaktori": {
        "name": "Reaktori",
        "operator": "compass",
        "cost_number": "0812",
    },
    "newton": {
        "name": "Newton",
        "operator": "jamix",
        "customer_id": "93077",
        "kitchen_id": "6",
        "menu_type_id": 56,
    },
    "konehuone": {
        "name": "Konehuone",
        "operator": "jamix",
        "customer_id": "93077",
        "kitchen_id": "6",
        "menu_type_id": 112,
    },
    "hertsi": {
        "name": "Hertsi",
        "operator": "sodexo",
        "kitchen_id": "111",
    },
}

ALIASES: dict[str, str] = {
    "food & co": "reaktori",
    "food and co": "reaktori",
    "foodco": "reaktori",
}


def _resolve_name(name: str) -> str | None:
    """Map a user-given name to a canonical restaurant key, or None."""
    key = name.strip().lower()
    if key in RESTAURANTS:
        return key
    return ALIASES.get(key)


# ── Compass Group (Reaktori) ─────────────────────────────────────

def _fetch_compass(cfg: dict) -> list[str]:
    url = (
        "https://www.compass-group.fi/menuapi/feed/json"
        f"?costNumber={cfg['cost_number']}&language=fi"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    today_str = date.today().isoformat()
    lines: list[str] = []

    for day in data.get("MenusForDays", []):
        day_date = (day.get("Date") or "")[:10]
        if day_date != today_str:
            continue
        for menu in day.get("SetMenus", []):
            name = menu.get("Name", "")
            components = menu.get("Components", [])
            if components:
                foods = ", ".join(components)
                lines.append(f"  {name}: {foods}")
            elif name:
                lines.append(f"  {name}")
        break

    return lines


# ── Jamix (Newton, Konehuone) ────────────────────────────────────

def _fetch_jamix(cfg: dict) -> list[str]:
    today = date.today()
    date_str = today.strftime("%Y%m%d")
    url = (
        f"https://fi.jamix.cloud/apps/menuservice/rest/haku/menu"
        f"/{cfg['customer_id']}/{cfg['kitchen_id']}"
        f"?lang=fi&date={date_str}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    target_mt = cfg.get("menu_type_id")
    lines: list[str] = []

    if not isinstance(data, list):
        return lines

    for kitchen_block in data:
        for menu_type in kitchen_block.get("menuTypes", []):
            mt_id = menu_type.get("menuTypeId")
            if target_mt is not None and mt_id != target_mt:
                continue
            for day_block in menu_type.get("menus", []):
                day_date = day_block.get("date", 0)
                menu_date = _jamix_ts_to_date(day_date)
                if menu_date != today:
                    continue
                for option in day_block.get("menuItems", []):
                    option_name = option.get("name", "")
                    rows = option.get("rows", [])
                    food_names = []
                    for row in rows:
                        rname = row.get("name", "")
                        diets = row.get("diets", "")
                        if rname:
                            entry = f"{rname} ({diets})" if diets else rname
                            food_names.append(entry)
                    if food_names:
                        lines.append(f"  {option_name}: {', '.join(food_names)}")
                    elif option_name:
                        lines.append(f"  {option_name}")
    return lines


def _jamix_ts_to_date(ts: int | str) -> date | None:
    """Jamix dates can be epoch-millis or YYYYMMDD ints."""
    try:
        ts_int = int(ts)
    except (ValueError, TypeError):
        return None
    if ts_int > 30000000:
        from datetime import datetime, timezone
        return datetime.fromtimestamp(ts_int / 1000, tz=timezone.utc).date()
    y = ts_int // 10000
    m = (ts_int % 10000) // 100
    d = ts_int % 100
    try:
        return date(y, m, d)
    except ValueError:
        return None


# ── Sodexo (Hertsi) ──────────────────────────────────────────────

def _fetch_sodexo(cfg: dict) -> list[str]:
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")
    url = (
        f"https://www.sodexo.fi/ruokalistat/output/daily_json"
        f"/{cfg['kitchen_id']}/{date_str}"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    lines: list[str] = []
    courses = data.get("courses", {})
    if isinstance(courses, dict):
        courses = courses.values()

    for course in courses:
        title = course.get("title_fi", "")
        diets = course.get("dietcodes", "")
        category = course.get("category", "")
        if title:
            entry = title
            if diets:
                entry += f" ({diets})"
            if category:
                entry = f"  [{category}] {entry}"
            else:
                entry = f"  {entry}"
            lines.append(entry)

    return lines


# ── Public API ───────────────────────────────────────────────────

_FETCHERS = {
    "compass": _fetch_compass,
    "jamix": _fetch_jamix,
    "sodexo": _fetch_sodexo,
}


def get_menu(restaurant: str) -> str:
    """Fetch today's lunch menu for *restaurant* and return readable text.

    Returns a user-friendly string, including an error message on failure.
    """
    key = _resolve_name(restaurant)
    if key is None:
        available = ", ".join(r["name"] for r in RESTAURANTS.values())
        return f"Tuntematon ravintola: '{restaurant}'. Vaihtoehdot: {available}"

    cfg = RESTAURANTS[key]
    fetcher = _FETCHERS.get(cfg["operator"])
    if fetcher is None:
        return f"Ei toteutusta operaattorille: {cfg['operator']}"

    try:
        lines = fetcher(cfg)
    except requests.RequestException as exc:
        print(f"[Menu] HTTP error for {cfg['name']}: {exc}", file=sys.stderr)
        return f"Ruokalistan haku epäonnistui ravintolalle {cfg['name']}."
    except Exception as exc:
        print(f"[Menu] Unexpected error for {cfg['name']}: {exc}", file=sys.stderr)
        return f"Ruokalistan haku epäonnistui ravintolalle {cfg['name']}."

    if not lines:
        return f"Ravintolalle {cfg['name']} ei löytynyt tämän päivän ruokalistaa."

    header = f"{cfg['name']} — {date.today().strftime('%d.%m.%Y')}:"
    return header + "\n" + "\n".join(lines)


def list_restaurants() -> list[str]:
    """Return the canonical names of all supported restaurants."""
    return [r["name"] for r in RESTAURANTS.values()]
