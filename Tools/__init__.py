"""Tools package: command parsing and local tool execution."""

from __future__ import annotations

from dataclasses import dataclass

from .commands import parse_command
from .motors import execute

# Lyhyet vastaukset käyttäjän kielellä (fi=en oletus)
_TOOL_RESPONSES: dict[tuple[str, str], str] = {
    ("play_ok", "fi"): "Soitetaan",
    ("play_ok", "en"): "Playing",
    ("queue_ok", "fi"): "Lisätty jonoon",
    ("queue_ok", "en"): "Added to queue",
    ("skip_ok", "fi"): "Seuraava kappale",
    ("skip_ok", "en"): "Next song",
    ("pause_ok", "fi"): "Tauko",
    ("pause_ok", "en"): "Paused",
    ("resume_ok", "fi"): "Jatketaan",
    ("resume_ok", "en"): "Resuming",
    ("stop_ok", "fi"): "Lopetettu",
    ("stop_ok", "en"): "Stopped",
    ("nothing_playing", "fi"): "Ei mitään soittamassa",
    ("nothing_playing", "en"): "Nothing playing",
    ("music_not_ready", "fi"): "Musiikkitoiminto ei ole valmis",
    ("music_not_ready", "en"): "Music not ready",
    ("greeting", "fi"): "Hei! Mitä haluat?",
    ("greeting", "en"): "Hi! What can I do for you?",
    ("time_prefix", "fi"): "Kello on",
    ("time_prefix", "en"): "The time is",
}


def _tr(key: str, lang: str) -> str:
    return _TOOL_RESPONSES.get((key, lang), _TOOL_RESPONSES.get((key, "fi"), key))


@dataclass
class ToolExecutionResult:
    handled: bool
    action: str | None = None
    response: str | None = None
    success: bool = True


def handle_speech(text: str, *, language: str = "") -> ToolExecutionResult:
    """Parse text and execute a local tool command when possible. language: fi, en, etc."""
    if not (text or "").strip():
        return ToolExecutionResult(handled=False)

    lang = "en" if language == "en" else "fi"
    cmd = parse_command(text.strip())
    if not cmd:
        return ToolExecutionResult(handled=False)

    action = cmd.get("action", "")
    response = cmd.get("response")
    success = True

    if action == "music_play":
        try:
            from .music import play_async, resolve_url, is_genre_like, check_music_ready
            if not check_music_ready():
                response = _tr("music_not_ready", lang)
                success = False
            else:
                query = (cmd.get("query") or "music").strip() or "music"
                url = resolve_url(query)
                if url is None:
                    response = "En löytänyt biisiä." if lang == "fi" else "I couldn't find a track."
                    success = False
                else:
                    play_async(query, url=url)
                    if is_genre_like(query):
                        response = "Soitetaan." if lang == "fi" else "Playing."
                    else:
                        response = cmd.get("response") or f"{_tr('play_ok', lang)}: {query}."
        except Exception:
            response = _tr("music_not_ready", lang)
            success = False
    elif action == "music_queue":
        try:
            from .music import add_to_queue
            add_to_queue(cmd.get("query", ""))
            response = cmd.get("response") or _tr("queue_ok", lang) + "."
        except Exception:
            response = _tr("music_not_ready", lang)
            success = False
    elif action == "music_skip":
        try:
            from .music import skip
            ok = skip()
            response = cmd.get("response") if ok else _tr("nothing_playing", lang) + "."
            if ok and not response:
                response = _tr("skip_ok", lang) + "."
        except Exception:
            response = _tr("music_not_ready", lang)
            success = False
    elif action == "music_pause":
        try:
            from .music import pause
            ok = pause()
            response = _tr("pause_ok", lang) + "." if ok else _tr("nothing_playing", lang) + "."
        except Exception:
            response = _tr("music_not_ready", lang)
            success = False
    elif action == "music_resume":
        try:
            from .music import resume
            ok = resume()
            response = _tr("resume_ok", lang) + "." if ok else _tr("nothing_playing", lang) + "."
        except Exception:
            response = _tr("music_not_ready", lang)
            success = False
    elif action == "music_stop":
        try:
            from .music import stop
            ok = stop()
            response = _tr("stop_ok", lang) + "." if ok else _tr("nothing_playing", lang) + "."
        except Exception:
            response = _tr("music_not_ready", lang)
            success = False
    elif action == "menu_check":
        from .menu import get_all_menus, get_menu

        restaurant = cmd.get("restaurant")
        if restaurant:
            response = get_menu(restaurant)
        else:
            response = get_all_menus()
        if response.startswith("Failed to fetch") or response.startswith("Unknown restaurant"):
            success = False
    elif action == "tell_time":
        from datetime import datetime

        now = datetime.now().strftime("%H:%M")
        response = f"{_tr('time_prefix', lang)} {now}."
    elif action == "tell_joke":
        import random

        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I told my robot a joke. It didn't laugh. Turns out it had no sense of humor, just sensors.",
            "Why was the robot so bad at soccer? It kept rebooting instead of shooting.",
            "What do you call a robot that takes the long way around? R2-Detour.",
        ]
        response = random.choice(jokes)
    elif action == "volume_up":
        from .music import volume_up

        vol = volume_up()
        response = f"Volume set to {vol}%."
    elif action == "volume_down":
        try:
            from .music import volume_down
            vol = volume_down()
            response = f"Ääni {vol}%."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis."
            success = False
    elif action == "greeting":
        response = _tr("greeting", lang)
    elif action in ("help", "acknowledgment"):
        pass
    else:
        execute(cmd)

    return ToolExecutionResult(
        handled=True,
        action=action or None,
        response=response,
        success=success,
    )
