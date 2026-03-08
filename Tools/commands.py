"""Parse recognized speech into a command dict or None.

Uses keyword-based (``in``) matching so that Whisper output like
"could you please play some beatles" still triggers the play command.

Priority order:
  1. Prefix commands (play, queue) -- checked first because they
     carry a query that may contain control words like "next" or
     "stop" (e.g. "play next level" should be play, not skip).
  2. Simple commands (skip, pause, resume, stop, volume, etc.)
  3. Menu / lunch commands with restaurant detection.
"""

# ── Keyword groups ────────────────────────────────────────────────

# Prefix keywords: the rest of the text after the keyword is the query.
# Longer prefixes first so "play music" is not eaten by bare "play".
PLAY_PREFIXES = (
    "put on some", "put on", "soita jotain", "soita musiikki", "soita",
    "play music", "play a song", "play song", "play some", "play me",
    "i want to hear", "i want to listen to", "i wanna hear",
    "let's listen to", "lets listen to",
    "can you play", "could you play",
    "play",
)
QUEUE_PREFIXES = (
    "add to queue", "add to the queue", "lisää jonoon", "lisää listaan",
    "queue up", "put in queue", "put in the queue",
    "enqueue", "queue",
)

# Simple keywords (no query).
SKIP_KEYWORDS = (
    "next song", "next track", "skip this song", "skip this track",
    "skip song", "skip track", "skip this", "skip it",
    "play next", "go to next", "move to next",
    "seuraava kappale", "seuraava biisi", "seuraava", "seuraavaksi",
    "skip", "next",
)
PAUSE_KEYWORDS = (
    "pause music", "pause the music", "pause song", "pause the song",
    "pause playback", "hold the music", "tauko", "pauseta", "pausetta",
    "pause",
)
RESUME_KEYWORDS = (
    "resume music", "resume the music", "resume song", "resume playback",
    "continue playing", "continue the music", "keep playing",
    "unpause", "un-pause", "jatka", "jatka musiikki", "jatka soitto",
    "resume",
)
STOP_KEYWORDS = (
    "stop music", "stop the music", "stop playing", "stop the song",
    "stop playback", "turn off music", "turn off the music",
    "kill the music", "cut the music", "silence",
    "lopeta", "lopeta musiikki", "lopeta soitto", "musiikki pois",
    "stop",
)
VOLUME_UP_KEYWORDS = (
    "turn it up", "volume up", "louder", "raise volume",
    "increase volume", "crank it up", "pump it up",
    "turn up the volume", "make it louder",
)
VOLUME_DOWN_KEYWORDS = (
    "turn it down", "volume down", "quieter", "lower volume",
    "decrease volume", "turn down the volume", "make it quieter",
    "not so loud", "too loud",
)

# Menu / lunch keywords — trigger a menu_check action.
MENU_KEYWORDS = (
    "lunch menu", "today's menu", "todays menu",
    "what's for lunch", "whats for lunch",
    "what's on the menu", "whats on the menu",
    "what are they serving", "what do they serve",
    "food menu", "daily menu", "check the menu",
    "what's cooking", "whats cooking",
    "ruokalista", "lounaslista", "päivän ruoka", "paivan ruoka",
    "päivän lounas", "paivan lounas", "mitä ruuaksi", "mita ruuaksi",
    "mitä lounaaksi", "mita lounaaksi", "mitä on ruokana", "mita on ruokana",
    "mitä on lounaalla", "mita on lounaalla", "mitä ruokana", "mita ruokana",
    "lunch", "menu",
)
RESTAURANT_ALIASES: dict[str, str] = {
    "reaktori": "reaktori",
    "reaktorin": "reaktori",
    "reaktorissa": "reaktori",
    "foodco": "reaktori",
    "food co": "reaktori",
    "food and co": "reaktori",
    "food & co": "reaktori",
    "konehuone": "konehuone",
    "konehuoneen": "konehuone",
    "konehuoneessa": "konehuone",
    "cafe konehuone": "konehuone",
    "café konehuone": "konehuone",
    "hertsi": "hertsi",
    "hertsin": "hertsi",
    "hertsissä": "hertsi",
    "hertsissa": "hertsi",
    "newton": "newton",
    "newtonin": "newton",
    "newtonissa": "newton",
}

# Greeting / small talk
GREETING_KEYWORDS = (
    "hello", "hey", "hi there", "howdy", "good morning",
    "good afternoon", "good evening", "what's up", "whats up",
    "how are you", "how's it going", "hows it going",
    "moi", "hei", "terve", "moro", "moikka", "päivää", "paivaa",
    "hi",
)

# Help
HELP_KEYWORDS = (
    "what can you do", "help me", "show commands",
    "what are your commands", "list commands",
    "what do you do", "help", "apua", "mitä osaat", "mita osaat",
)

# Time
TIME_KEYWORDS = (
    "what time is it", "what's the time", "whats the time",
    "tell me the time", "current time",
    "paljonko kello on", "mitä kello on", "mita kello on",
    "mitä kello", "mita kello", "kello",
    "time",
)

# Joke
JOKE_KEYWORDS = (
    "tell me a joke", "say something funny", "make me laugh",
    "got a joke", "do you know a joke", "know any jokes",
    "kerro vitsi", "vitsi", "vitsiä", "vitsia",
    "joke",
)


import re


def _extract_after(text: str, keyword: str) -> str:
    """Return everything after *keyword* in *text* (case-insensitive)."""
    idx = text.lower().find(keyword)
    if idx == -1:
        return ""
    return text[idx + len(keyword):].strip()


def _word_match(keyword: str, text: str) -> bool:
    """True if *keyword* appears as whole words in *text* (not inside another word)."""
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text))


def parse_command(text: str) -> dict | None:
    """Parse user text into an action dict, or None if not a known command."""
    if not text or not text.strip():
        return None
    normalized = text.strip().lower()

    # ── 1. Resume checked before play so "continue playing" is not
    #        eaten by the "play" prefix ────────────────────────────

    if any(_word_match(kw, normalized) for kw in RESUME_KEYWORDS):
        return {"action": "music_resume", "response": "Resuming playback."}

    # ── 2. Prefix commands (carry a query) — checked before simple
    #        skip/pause/stop so "play next level" is play, not skip

    for prefix in QUEUE_PREFIXES:
        if _word_match(prefix, normalized):
            query = _extract_after(text.strip(), prefix)
            if not query:
                continue
            return {
                "action": "music_queue",
                "query": query,
                "response": f"Added to queue: {query}",
            }

    for prefix in PLAY_PREFIXES:
        if _word_match(prefix, normalized):
            query = _extract_after(text.strip(), prefix)
            if not query:
                query = "music"
            return {
                "action": "music_play",
                "query": query,
                "response": f"Playing: {query}",
            }

    # ── 3. Simple music commands ──────────────────────────────────

    if any(_word_match(kw, normalized) for kw in SKIP_KEYWORDS):
        return {"action": "music_skip", "response": "Skipping to next song."}

    if any(_word_match(kw, normalized) for kw in PAUSE_KEYWORDS):
        return {"action": "music_pause", "response": "Music paused."}

    if any(_word_match(kw, normalized) for kw in STOP_KEYWORDS):
        return {"action": "music_stop", "response": "Playback stopped."}

    # ── 3. Volume ─────────────────────────────────────────────────

    if any(_word_match(kw, normalized) for kw in VOLUME_UP_KEYWORDS):
        return {"action": "volume_up", "response": "Turning it up."}

    if any(_word_match(kw, normalized) for kw in VOLUME_DOWN_KEYWORDS):
        return {"action": "volume_down", "response": "Turning it down."}

    # ── 4. Menu / lunch commands ─────────────────────────────────

    if any(_word_match(kw, normalized) for kw in MENU_KEYWORDS):
        restaurant = _detect_restaurant(normalized)
        return {
            "action": "menu_check",
            "restaurant": restaurant,
        }

    # ── 5. Utility / conversational ───────────────────────────────

    if any(_word_match(kw, normalized) for kw in HELP_KEYWORDS):
        return {
            "action": "help",
            "response": (
                "I can play music, check lunch menus for Reaktori, Newton, "
                "Konehuone and Hertsi, tell you the time, and tell jokes. "
                "Just ask!"
            ),
        }

    if any(_word_match(kw, normalized) for kw in TIME_KEYWORDS):
        return {"action": "tell_time"}

    if any(_word_match(kw, normalized) for kw in JOKE_KEYWORDS):
        return {"action": "tell_joke"}

    if any(_word_match(kw, normalized) for kw in GREETING_KEYWORDS):
        return {"action": "greeting", "response": "Hey there! What can I do for you?"}

    return None


def _detect_restaurant(text: str) -> str | None:
    """Return the first restaurant name found in *text*, or None."""
    for alias, canonical in sorted(RESTAURANT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text:
            return canonical
    return None
