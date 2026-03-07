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
    else:
        execute(cmd)

    if response:
        say(response)
