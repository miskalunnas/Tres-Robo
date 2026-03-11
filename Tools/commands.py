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
# Longer prefixes first. Vain selkeät musiikkikomennot — ei "play" tai "laitetaan" yksin.
PLAY_PREFIXES = (
    "put on some", "put on", "put some music on", "throw on some", "throw on",
    "soita jotain", "soita musiikki", "soita", "soittakaa", "soittaisitko",
    "voisitko soittaa", "voisit soittaa", "voisko soittaa", "voisitko laittaa",
    "laita soimaan", "laitetaan jotain", "laitetaan", "pistä soimaan", "pistä soiton",
    "taustamusiikkia", "taustamusiikki", "taustalle", "jotain taustalle",
    "jotain musiikkia", "musiikkia", "vähän musiikkia", "vahan musiikkia",
    "play music", "play a song", "play song", "play some", "play me", "play a bit of",
    "play something", "play anything", "start some music", "start the music",
    "can you put on", "could you play", "could you put on",
    "hit play", "turn on some music", "turn on the music",
)
# Ei pelkkä "play" tai "queue" — liian herkkä.
QUEUE_PREFIXES = (
    "add to queue", "add to the queue", "add this to queue", "add that to queue",
    "lisää jonoon", "lisää listaan", "lisää tämä jonoon", "laita jonoon",
    "laita seuraavaksi", "seuraavaksi soita", "seuraavaksi tämä", "soita tämä seuraavaksi",
    "queue up", "queue this", "put in queue", "put in the queue", "put this in queue",
    "enqueue", "add next", "play that next", "next up", "after this play",
)
# Täytesanat: jos query on vain näitä, älä soita (ei "laitetaan vaikka" tms.)
PLAY_QUERY_BLOCKLIST = frozenset({"vaikka", "sitten", "nyt", "vähän", "vahan", "ehkä", "ehka"})
# Pelkkä genre = soita genren mukaista (Whisper voi tunnistaa "jazz" suoraan)
GENRE_ONLY_WORDS = frozenset({"jazz", "chill", "lo-fi", "lofi", "rauhallinen", "rento", "taustamusiikki"})

# Simple keywords (no query).
# "seuraavaksi" ja "skip" voivat esiintyä keskustelussa ("arvaa mitä rakennetaan seuraavaksi") — ei triggeröidä pitkissä lauseissa.
AMBIGUOUS_SKIP_KEYWORDS = frozenset({"seuraavaksi", "skip"})
SKIP_KEYWORDS = (
    "next song", "next track", "next one", "next tune", "another song", "different song",
    "skip this song", "skip this track", "skip this", "skip it", "skip to next",
    "skip song", "skip track", "change song", "change the song", "switch song",
    "play next", "go to next", "move to next", "on to the next",
    "seuraava kappale", "seuraava biisi", "seuraavaksi",
    "vaihda biisi", "vaihda kappale", "toinen biisi", "toinen kappale", "uusi biisi",
    "ei tätä", "not this one", "not this song",
    "skip",
)
PAUSE_KEYWORDS = (
    "pause music", "pause the music", "pause song", "pause the song",
    "pause playback", "hold the music", "freeze the music",
    "tauko", "pauseta", "pausetta", "pysäytä", "pysäytä musiikki", "pidä tauko",
    "pause",
)
RESUME_KEYWORDS = (
    "resume music", "resume the music", "resume song", "resume playback",
    "continue playing", "continue the music", "keep playing",
    "unpause", "un-pause", "play again", "start again", "start playing again",
    "jatka", "jatka musiikki", "jatka soitto", "jatka soittaminen",
    "alkaa soittaa", "soita taas", "soita uudestaan",
    "resume",
)
STOP_KEYWORDS = (
    "stop music", "stop the music", "stop playing", "stop the song",
    "stop playback", "turn off music", "turn off the music", "switch off music",
    "kill the music", "cut the music", "silence", "sammuta musiikki",
    "stop it", "stop that", "no more music", "music off", "that's enough",
    "enough music", "shut off the music", "turn off the song",
    "lopeta", "lopeta musiikki", "lopeta soitto", "lopeta se", "lopeta tuo",
    "ei enää musiikkia", "ei musiikkia", "musiikki pois",
    "poista kaikki", "poista kaikki musiikki", "tyhjennä jono", "tyhjennä lista",
    "clear", "clear all", "clear queue", "clear the queue",
    "stop",
)
VOLUME_UP_KEYWORDS = (
    "turn it up", "volume up", "volume upp", "louder", "raise volume",
    "increase volume", "crank it up", "pump it up", "boost volume",
    "turn up the volume", "make it louder", "a bit louder", "little louder",
    "äänenvoimakkuus ylös", "ääntä ylös", "ääni ylös", "kovemmalle", "kovempaa",
    "kovemmaksi", "ääni kovemmaksi", "ääntä kovemmaksi",
)
VOLUME_DOWN_KEYWORDS = (
    "turn it down", "volume down", "quieter", "lower volume", "softer",
    "decrease volume", "turn down the volume", "make it quieter",
    "not so loud", "too loud", "a bit quieter", "little quieter",
    "äänenvoimakkuus alas", "ääntä alas", "ääni alas", "hiljemmalle", "hiljempaa",
    "hiljemmaksi", "ääni hiljemmaksi", "ääntä hiljemmaksi",
)

# Menu / lunch keywords — trigger a menu_check action.
MENU_KEYWORDS = (
    "lunch menu", "today's menu", "todays menu",
    "what's for lunch", "whats for lunch",
    "what's on the menu", "whats on the menu",
    "what are they serving", "what do they serve",
    "food menu", "daily menu", "check the menu",
    "what's cooking", "whats cooking",
    "what's for food", "whats for food", "what food", "what to eat",
    "ruokalista", "lounaslista", "päivän ruoka", "paivan ruoka",
    "päivän lounas", "paivan lounas", "mitä ruuaksi", "mita ruuaksi",
    "mitä lounaaksi", "mita lounaaksi", "mitä on ruokana", "mita on ruokana",
    "mitä on lounaalla", "mita on lounaalla", "mitä ruokana", "mita ruokana",
    "mitä ruokaa", "mita ruokaa", "mitä ruokaa on", "mita ruokaa on",
    "mitä ruokaa tänään", "mita ruokaa tanaan", "mitä tänään ruokana", "mita tanaan ruokana",
    "mitä on ruokaa", "mita on ruokaa", "mitä housella ruokana", "mita housella ruokana",
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
    "miten menee", "mitä kuuluu", "mita kuuluu",
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

# Acknowledgment / kiitokset — lyhyt vastaus, ei LLM
ACKNOWLEDGMENT_KEYWORDS = (
    "kiitos kun katsoit", "kiitos paljon", "kiitos siitä",
)


import re
from difflib import SequenceMatcher

# Estää vääriä osumia: "don't stop" ≠ stop, "hold on a sec" ≠ pause
_STOP_BLOCKLIST = re.compile(r"\b(?:don'?t|do\s+not)\s+stop\b", re.I)
_PAUSE_BLOCKLIST = re.compile(r"\bhold\s+on\s+(?:a\s+)?(?:sec|moment|minute)\b", re.I)

# Whisper-virheet: korjataan ennen parsintaa (esim. "jas" → "jazz").
# Avain = väärin tunnistettu, arvo = oikea muoto. Käytetään word-boundary korvauksia.
_WHISPER_CORRECTIONS: dict[str, str] = {
    "jas": "jazz",
    "jassi": "jazz",
    "jass": "jazz",
    "shill": "chill",
    "seuraa": "seuraava",
}


def _normalize_whisper_text(text: str) -> str:
    """Korjaa yleisiä Whisper-virheitä ennen komentojen parsintaa."""
    if not text or not text.strip():
        return text
    result = text
    for wrong, correct in _WHISPER_CORRECTIONS.items():
        # Korvaa vain kokonaisina sanoina (ei "jasmine" → "jazzmine")
        result = re.sub(rf"\b{re.escape(wrong)}\b", correct, result, flags=re.IGNORECASE)
    return result


def _extract_after(text: str, keyword: str) -> str:
    """Return everything after *keyword* in *text* (case-insensitive)."""
    idx = text.lower().find(keyword)
    if idx == -1:
        return ""
    return text[idx + len(keyword):].strip()


def _word_match(keyword: str, text: str) -> bool:
    """True if *keyword* appears as whole words in *text* (not inside another word)."""
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text, re.IGNORECASE))


def _fuzzy_match_keywords(keywords: tuple[str, ...], text: str, *, min_ratio: float = 0.82) -> bool:
    """True if any word in text is similar enough to any keyword (Whisper-virheet: skipp→skip)."""
    words = re.findall(r"[a-zäöå]+", text.lower())
    for w in words:
        if len(w) < 3:
            continue
        for kw in keywords:
            if len(kw) < 3:
                continue
            ratio = SequenceMatcher(None, w, kw).ratio()
            if ratio >= min_ratio:
                return True
    return False


def parse_command(text: str) -> dict | None:
    """Parse user text into an action dict, or None if not a known command."""
    if not text or not text.strip():
        return None
    text = _normalize_whisper_text(text)
    normalized = text.strip().lower()

    # ── 0. Pelkkä genre (esim. "jazz", "chill") = soita
    if normalized in GENRE_ONLY_WORDS:
        return {"action": "music_play", "query": normalized}

    # ── 1. Resume checked before play so "continue playing" is not
    #        eaten by the "play" prefix ────────────────────────────

    if any(_word_match(kw, normalized) for kw in RESUME_KEYWORDS) or _fuzzy_match_keywords(RESUME_KEYWORDS, normalized):
        return {"action": "music_resume", "response": "Jatketaan."}

    # ── 2. Prefix commands (carry a query) — checked before simple
    #        skip/pause/stop so "play next level" is play, not skip

    for prefix in QUEUE_PREFIXES:
        if _word_match(prefix, normalized):
            query = _extract_after(text.strip(), prefix)
            if not query or query.lower().strip() in PLAY_QUERY_BLOCKLIST:
                continue
            return {
                "action": "music_queue",
                "query": query,
                "response": f"Lisätty jonoon: {query}",
            }

    for prefix in PLAY_PREFIXES:
        if _word_match(prefix, normalized):
            query = _extract_after(text.strip(), prefix)
            if not query:
                query = "music"
            # Älä triggeröi jos query on vain täytesana ("laitetaan vaikka" jne.)
            if query.lower().strip() in PLAY_QUERY_BLOCKLIST:
                continue
            return {"action": "music_play", "query": query}

    # ── 3. Simple music commands ──────────────────────────────────
    # Fuzzy fallback: Whisper-virheet (skipp→skip, pauseta→pause) — difflib 0.82 kynnys.

    skip_matched = [kw for kw in SKIP_KEYWORDS if _word_match(kw, normalized)]
    fuzzy_skip = _fuzzy_match_keywords(SKIP_KEYWORDS, normalized)
    if skip_matched or fuzzy_skip:
        words = re.findall(r"\w+", normalized)
        # Pitkä lause: vain "seuraavaksi"/"skip" = keskustelua, ei komento
        if len(words) > 4:
            strong_match = any(kw not in AMBIGUOUS_SKIP_KEYWORDS for kw in skip_matched)
            if not strong_match and not skip_matched and fuzzy_skip:
                pass  # fuzzy match pitkässä lauseessa → älä triggeröi
            elif strong_match:
                return {"action": "music_skip", "response": "Skipping to next song."}
        else:
            return {"action": "music_skip", "response": "Skipping to next song."}

    if any(_word_match(kw, normalized) for kw in PAUSE_KEYWORDS) or _fuzzy_match_keywords(PAUSE_KEYWORDS, normalized):
        if not _PAUSE_BLOCKLIST.search(text):
            return {"action": "music_pause", "response": "Tauko."}

    if any(_word_match(kw, normalized) for kw in STOP_KEYWORDS) or _fuzzy_match_keywords(STOP_KEYWORDS, normalized):
        if not _STOP_BLOCKLIST.search(text):
            return {"action": "music_stop", "response": "Lopetettu."}

    # ── 3. Volume ─────────────────────────────────────────────────

    if any(_word_match(kw, normalized) for kw in VOLUME_UP_KEYWORDS):
        return {"action": "volume_up", "response": "Turning it up."}

    if any(_word_match(kw, normalized) for kw in VOLUME_DOWN_KEYWORDS):
        return {"action": "volume_down", "response": "Hiljemmalle."}

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
                "Voin soittaa musiikkia (soita, skip, tauko, jatka, lopeta), "
                "tarkistaa ruokalistat, kertoa kellonajan ja vitsin. Kysy vapaasti."
            ),
        }

    if any(_word_match(kw, normalized) for kw in TIME_KEYWORDS):
        return {"action": "tell_time"}

    if any(_word_match(kw, normalized) for kw in JOKE_KEYWORDS):
        return {"action": "tell_joke"}

    if any(_word_match(kw, normalized) for kw in GREETING_KEYWORDS):
        return {"action": "greeting", "response": "Hei! Mitä haluat?"}

    if any(_word_match(kw, normalized) for kw in ACKNOWLEDGMENT_KEYWORDS):
        return {"action": "acknowledgment", "response": "Eipä kestä!"}

    return None


def _detect_restaurant(text: str) -> str | None:
    """Return the first restaurant name found in *text*, or None."""
    for alias, canonical in sorted(RESTAURANT_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in text:
            return canonical
    return None
