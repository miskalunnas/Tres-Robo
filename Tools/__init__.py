"""Tools package: command parsing, TTS, motors, music, etc."""

from tools.commands import parse_command
from tools.tts import say
from tools.motors import execute


def handle_speech(text: str) -> None:
    """
    Handle recognized speech: parse command, then run the appropriate tool
    (music, motors, etc.) and optionally speak a response.
    """
    if not (text or "").strip():
        return
    cmd = parse_command(text.strip())
    if not cmd:
        return

    action = cmd.get("action", "")
    response = cmd.get("response")

    if action == "music_play":
        from tools.music import play
        play(cmd.get("query", ""))
    elif action == "music_queue":
        from tools.music import add_to_queue
        add_to_queue(cmd.get("query", ""))
    elif action == "music_skip":
        from tools.music import skip
        skip()
    elif action == "music_pause":
        from tools.music import pause
        pause()
    elif action == "music_resume":
        from tools.music import resume
        resume()
    elif action == "music_stop":
        from tools.music import stop
        stop()
    elif action == "menu_check":
        from tools.menu import get_menu, get_all_menus
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
            "I told my robot a joke. It didn't laugh. Turns out it had no sense of humor — just sensors.",
            "Why was the robot so bad at soccer? It kept rebooting instead of shooting.",
            "What do you call a robot that takes the long way around? R2-Detour.",
        ]
        response = random.choice(jokes)
    elif action in ("volume_up", "volume_down", "greeting", "help"):
        pass
    else:
        execute(cmd)

    if response:
        say(response)
