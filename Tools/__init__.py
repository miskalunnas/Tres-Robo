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
        from .music import play

        success = play(cmd.get("query", ""))
        if not success:
            response = "I couldn't start playback."
    elif action == "music_queue":
        from .music import add_to_queue

        add_to_queue(cmd.get("query", ""))
    elif action == "music_skip":
        from .music import skip

        skip()
    elif action == "music_pause":
        from .music import pause

        pause()
    elif action == "music_resume":
        from .music import resume

        resume()
    elif action == "music_stop":
        from .music import stop

        stop()
    elif action == "menu_check":
        from .menu import get_all_menus, get_menu

        restaurant = cmd.get("restaurant")
        if restaurant:
            response = get_menu(restaurant)
        else:
            response = get_all_menus()
    elif action == "tell_time":
        from datetime import datetime

        now = datetime.now().strftime("%H:%M")
        response = f"It's {now}."
    elif action == "tell_joke":
        import random

        jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "I told my robot a joke. It didn't laugh. Turns out it had no sense of humor, just sensors.",
            "Why was the robot so bad at soccer? It kept rebooting instead of shooting.",
            "What do you call a robot that takes the long way around? R2-Detour.",
        ]
        response = random.choice(jokes)
    elif action in ("volume_up", "volume_down", "greeting", "help"):
        pass
    else:
        execute(cmd)

    return ToolExecutionResult(
        handled=True,
        action=action or None,
        response=response,
        success=success,
    )
