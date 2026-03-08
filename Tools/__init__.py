"""Tools package: command parsing and local tool execution."""

from __future__ import annotations

from dataclasses import dataclass

from .commands import parse_command
from .motors import execute


@dataclass
class ToolExecutionResult:
    handled: bool
    action: str | None = None
    response: str | None = None
    success: bool = True


def handle_speech(text: str) -> ToolExecutionResult:
    """Parse text and execute a local tool command when possible."""
    if not (text or "").strip():
        return ToolExecutionResult(handled=False)

    cmd = parse_command(text.strip())
    if not cmd:
        return ToolExecutionResult(handled=False)

    action = cmd.get("action", "")
    response = cmd.get("response")
    success = True

    if action == "music_play":
        try:
            from .music import play_async
            query = (cmd.get("query") or "music").strip() or "music"
            play_async(query)
            response = cmd.get("response") or f"Soitetaan: {query}."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis. Asenna yt-dlp ja ffmpeg tai mpv."
            success = False
    elif action == "music_queue":
        try:
            from .music import add_to_queue
            add_to_queue(cmd.get("query", ""))
            response = cmd.get("response") or "Lisätty jonoon."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis."
            success = False
    elif action == "music_skip":
        try:
            from .music import skip
            ok = skip()
            response = cmd.get("response") if ok else "Ei mitään soittamassa."
            if ok and not response:
                response = "Seuraava kappale."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis."
            success = False
    elif action == "music_pause":
        try:
            from .music import pause
            ok = pause()
            response = "Tauko." if ok else "Ei mitään pausettavana."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis."
            success = False
    elif action == "music_resume":
        try:
            from .music import resume
            ok = resume()
            response = "Jatketaan." if ok else "Ei mitään jatkettavana."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis."
            success = False
    elif action == "music_stop":
        try:
            from .music import stop
            ok = stop()
            response = "Lopetettu." if ok else "Ei mitään soittanut."
        except Exception:
            response = "Musiikkitoiminto ei ole valmis."
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
        response = f"Kello on {now}."
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
    elif action in ("greeting", "help", "acknowledgment"):
        pass
    else:
        execute(cmd)

    return ToolExecutionResult(
        handled=True,
        action=action or None,
        response=response,
        success=success,
    )
