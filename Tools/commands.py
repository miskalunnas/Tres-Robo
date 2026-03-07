"""Parse recognized speech into a command dict or None.

Uses keyword-based (``in``) matching so that Vosk output like
"please play some beatles" still triggers the play command.

Priority order:
  1. Prefix commands (play, queue) -- checked first because they
     carry a query that may contain control words like "next" or
     "stop" (e.g. "play next level" should be play, not skip).
  2. Simple commands (skip, pause, resume, stop) -- checked only
     when no prefix command matched.
"""

# ── Keyword groups ────────────────────────────────────────────────

# Prefix keywords: the rest of the text after the keyword is the query.
# Longer prefixes first so "play music" is not eaten by bare "play".
PLAY_PREFIXES = ("play music", "play song", "play some", "play")
QUEUE_PREFIXES = ("add to queue", "queue up", "queue")

# Simple keywords (no query).
SKIP_KEYWORDS = ("next song", "next track", "skip song", "skip track", "skip", "next")
PAUSE_KEYWORDS = ("pause music", "pause song", "pause")
RESUME_KEYWORDS = ("resume music", "resume song", "resume", "continue playing", "unpause")
STOP_KEYWORDS = ("stop music", "stop playing", "stop song", "stop the music", "stop")


def _extract_after(text: str, keyword: str) -> str:
    """Return everything after *keyword* in *text* (case-insensitive)."""
    idx = text.lower().find(keyword)
    if idx == -1:
        return ""
    return text[idx + len(keyword):].strip()


def parse_command(text: str) -> dict | None:
    """Parse user text into an action dict, or None if not a known command."""
    if not text or not text.strip():
        return None
    normalized = text.strip().lower()

    # ── 1. Prefix commands (carry a query) ────────────────────────
    # Checked first: "play next level" -> play with query "next level",
    # not a skip command.

    for prefix in QUEUE_PREFIXES:
        if prefix in normalized:
            query = _extract_after(text.strip(), prefix)
            if not query:
                continue
            return {
                "action": "music_queue",
                "query": query,
                "response": f"Added to queue: {query}",
            }

    for prefix in PLAY_PREFIXES:
        if prefix in normalized:
            query = _extract_after(text.strip(), prefix)
            if not query:
                query = "music"
            return {
                "action": "music_play",
                "query": query,
                "response": f"Playing: {query}",
            }

    # ── 2. Simple commands (no query) ─────────────────────────────
    # Only reached when text does NOT contain play/queue prefixes.

    if any(kw in normalized for kw in SKIP_KEYWORDS):
        return {"action": "music_skip", "response": "Skipping to next song."}

    if any(kw in normalized for kw in PAUSE_KEYWORDS):
        return {"action": "music_pause", "response": "Music paused."}

    if any(kw in normalized for kw in RESUME_KEYWORDS):
        return {"action": "music_resume", "response": "Resuming playback."}

    if any(kw in normalized for kw in STOP_KEYWORDS):
        return {"action": "music_stop", "response": "Playback stopped."}

    return None
