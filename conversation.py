"""Conversation orchestrator — sits between the audio pipeline and the brain.

State machine:
  OFFLINE: listens for wake word, ignores everything else.
  ONLINE:  passes every utterance to the LLM and speaks the reply.
           Returns to OFFLINE on inactivity timeout or a clear end-of-chat intent.
"""
import json
import re
import sys
import threading
import time

from brain import Brain

# Fallback: tunnista kieli tekstistä kun Whisper ei palauta
_ENGLISH_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did will would could should may might must shall can need "
    "what when where who which why how play stop skip pause resume time joke menu lunch volume hello hi hey yeah yes no ok okay cool thanks tell me about "
    "next song something music see look wearing tired happy room table other see you later"
.split()
)
# Lyhyet englanninkieliset fraasit (esim. "hey bot", "play something", "what do you see")
_ENGLISH_PHRASE_PATTERNS = (
    re.compile(r"\b(?:hey|hi|hello|yo)\s+(?:bot|founder)\b", re.I),
    re.compile(r"\b(?:play|skip|pause|resume|stop)\s+\w*\b", re.I),
    re.compile(r"\b(?:what|how|when|where|why)\s+\w+\b", re.I),
    re.compile(r"\b(?:tell me|give me|show me)\b", re.I),
    re.compile(r"\b(?:i (?:am|want|need|like))\b", re.I),
    re.compile(r"\b(?:what do you see|who is (?:there|here)|do i look)\b", re.I),
    re.compile(r"\b(?:volume up|volume down|turn (?:up|down))\b", re.I),
    re.compile(r"\b(?:next|previous)\s*(?:song|track)?\b", re.I),
    re.compile(r"\b(?:how are you|what('s| is) up)\b", re.I),
)


def _infer_language_from_text(text: str) -> str:
    """Yksinkertainen arvio: jos teksti sisältää en-tyyppisiä sanoja eikä ä/ö, → en."""
    if not text or len(text) < 2:
        return ""
    t = text.lower().strip()
    has_finnish_chars = "ä" in t or "ö" in t
    words = set(re.findall(r"\b[a-zäö]+\b", t))
    english_count = len(words & _ENGLISH_WORDS)
    if has_finnish_chars:
        return "fi"
    if english_count >= 2 or (english_count >= 1 and len(words) <= 4):
        return "en"
    if any(p.search(t) for p in _ENGLISH_PHRASE_PATTERNS):
        return "en"
    # Yksi selvä englanninkielinen komento (play, skip, next, stop, ...) → en
    if len(words) <= 2 and words & _ENGLISH_WORDS:
        return "en"
    return ""
from memory import MemoryStore
from Tools import handle_speech as handle_tool_speech
from Tools.commands import parse_command
from voice.tts import SpeechHandle, interrupt as interrupt_speech, speak

WAKE_WORDS = [ 
    "founderbot",
    "founderbott",
    "founder bot",
    "founder, bot",
    "found a bot",
    "founder bott",
    "founderbotti",
    "bot",
    "robot",
    "robotti",
    "botti",
    "founder",
]


def _normalize_for_wake(text: str) -> str:
    """Normalize text for wake word matching: lowercase, punctuation -> space, collapse spaces."""
    s = text.lower().strip()
    s = re.sub(r"[,.!?:;]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()
SESSION_END_PATTERNS = (
    re.compile(r"\b(?:goodbye|bye(?: bye)?|näkemiin|hei hei|moi moi)\b"),
    re.compile(
        r"\b(?:see you(?: later)?|talk to you later|catch you later|talk later)\b"
    ),
    re.compile(
        r"\b(?:go offline|go idle|go to sleep|sleep now|stop listening|you can stop listening|you can sleep|you can go to sleep)\b"
    ),
    # Nukkumaan / lepotila: botti menee sleep-tilaan, ei reagoi ennen herätyssanaa
    re.compile(
        r"\b(?:mene lepotilaan|siirry lepotilaan|mene nukkumaan|voit mennä nukkumaan|voit mennä lepotilaan|mennä nukkumaan|nuku nyt|mene nukkumaan nyt|lepotila|mene lepoon)\b"
    ),
    # Lepotila: "ole hiljaa", "älä puhu" jne.
    re.compile(
        r"\b(?:ole nyt hiljaa|ole hiljaa|älä puhu|voit olla hiljaa|ei tarvitse vastata|pysy hiljaa)\b"
    ),
    re.compile(
        r"\b(?:be quiet now|stay quiet|don't talk|shut up now|no need to respond|just be quiet)\b"
    ),
    re.compile(
        r"\b(?:that's all(?: for now)?|that is all(?: for now)?|that's everything|that is everything|we(?: are|'re) done|i(?: am|'m) done(?: for now)?|done for now|nothing else|no more questions|all good now)\b"
    ),
    re.compile(
        r"\b(?:siinä kaikki|tässä kaikki|ei muuta|ei muuta tällä erää|ollaan valmiita|se oli siinä|palataan myöhemmin|jutellaan myöhemmin|puhutaan myöhemmin|jatketaan myöhemmin)\b"
    ),
    # "Lopetetaan puhuminen" — kun käyttäjä haluaa lopettaa keskustelun
    re.compile(
        r"\b(?:lopetetaan|lopetetaan puhuminen|lopetetaan tää|lopetetaan tämä|sopii näin|selvä kiitos|okei kiitos|okei selvä|kiitos hei|kiitos moi|kiitos näkemiin|ei tarvitse enää|en tarvitse enää|se siinä)\b"
    ),
)
INTERRUPT_WORDS = [
    "stop",
    "stop talking",
    "be quiet",
    "quiet",
    "pause",
    "wait",
    "hold on",
    "shut up",
    "hiljaa",
    "tuki",
    "lopeta puhuminen",
    "hetki",
]

# Puhe on suunnattu botille (ei taustakeskustelu tai toiselle henkilölle).
# Käytetään _looks_like_clear_interrupt:ssa — suodattaa "mitä sanoit siitä kokouksesta" jne.
ADDRESSING_KEYWORDS = (
    "bot", "botti", "robot", "founderbot", "robotti",
    "soita", "play", "kerro", "lopeta", "tauko", "jatka", "seuraava", "skip", "pause", "resume", "stop",
    "voisitko", "voisit", "could you", "can you", "would you", "tee", "laita", "lisää",
    "paljonko kello", "mitä kello", "aika", "time",
    "vitsi", "joke", "ruokalista", "menu", "lunch",
    "ääni", "volume", "kovemmalle", "hiljemmalle", "louder", "quieter",
)

INACTIVITY_TIMEOUT = 40.0  # seconds of silence before going offline


class ConversationEngine:
    wake_word = WAKE_WORDS[0]

    def is_online(self) -> bool:
        return self._online

    def __init__(self) -> None:
        self._store = MemoryStore()
        self._store.ensure_knowledge_loaded()
        self._brain = Brain(self._store)
        self._online = False
        self._last_activity = time.monotonic()
        self._session_id: str | None = None
        self._person_id: str | None = None
        self._lock = threading.RLock()
        self._reply_cancel_event: threading.Event | None = None
        self._startup_vision_ready: threading.Event | None = None
        self._reply_generation = 0
        self._active_reply_text = ""
        self._last_spoken_text = ""
        self._session_language: str = ""  # viimeisin tunnistettu kieli

    # ------------------------------------------------------------------
    def handle(self, text: str, now: float, *, language: str = "") -> None:
        """Called by main.py for every transcribed utterance."""
        with self._lock:
            if self._online and (now - self._last_activity) >= INACTIVITY_TIMEOUT:
                print(
                    f"[Engine] No speech for {INACTIVITY_TIMEOUT:.0f}s — going OFFLINE."
                )
                self._end_session("timeout")

            normalized = text.lower()
            normalized_flex = _normalize_for_wake(text)
            matched_wake_word = next(
                (w for w in WAKE_WORDS if w in normalized or w in normalized_flex), None
            )

            if not self._online:
                print(f"[Offline heard] {text}")
                if not matched_wake_word:
                    return
                remainder = self._strip_phrase(text, matched_wake_word)
                has_followup = bool(remainder.strip())
                if has_followup:
                    self._start_session(now, matched_wake_word, announce=False)
                    self._process_online_text(remainder.strip(), now=now, language=language)
                else:
                    self._start_session(now, matched_wake_word, announce=False)
                    self._speak_reply("Hei!")
                return

            self._process_online_text(text, now=now, language=language)

    def handle_interruption(self, text: str, now: float, *, language: str = "") -> bool:
        """Handle an utterance captured while the bot is speaking.
        Botti kuuntelee koko ajan; text = alusta lähtien kuunneltu puhe (koko segmentti).
        Päätetään reagoidaanko: vain selkeä keskeytys (wake word, stop, pitkä tarkoituksellinen lause)."""
        with self._lock:
            normalized = text.lower()
            normalized_flex = _normalize_for_wake(text)
            matched_wake_word = next(
                (w for w in WAKE_WORDS if w in normalized or w in normalized_flex), None
            )
            matched_interrupt = next((word for word in INTERRUPT_WORDS if word in normalized), None)
            matched_session_end = self._is_session_end_intent(text)
            local_command = parse_command(text)
            # Lyhyet segmentit: 1–3 sanaa ilman wake/interrupt/session_end/komento → taustamelu, hylätään
            words = re.findall(r"\w+", normalized)
            if (
                not matched_wake_word
                and not matched_interrupt
                and not matched_session_end
                and local_command is None
                and len(words) <= 3
            ):
                preview = (text[:40] + "...") if len(text) > 40 else text
                print(f"[Interrupt ignored] Short (no wake/command): {preview}")
                return False
            clear_interrupt = self._looks_like_clear_interrupt(text)
            if matched_session_end:
                print(f"[Interrupt heard] {text}")
                self._cancel_active_reply_locked()
                interrupt_speech()
                self._last_activity = now
                self._speak_reply("Okay, going offline.", end_session_reason="goodbye")
                return True
            if (
                not matched_wake_word
                and not matched_interrupt
                and local_command is None
                and not clear_interrupt
            ):
                # Ei tarkoituksellinen keskeytys — botti jatkaa normaalisti, ei keskeytä
                preview = (text[:50] + "...") if len(text) > 50 else text
                print(f"[Interrupt ignored] {preview}")
                return False

            if matched_wake_word:
                remainder = self._strip_phrase(text, matched_wake_word)
            elif matched_interrupt:
                remainder = self._strip_phrase(text, matched_interrupt)
            else:
                remainder = text.strip()

            is_echo = bool(remainder and self._looks_like_echo(remainder))

            print(f"[Interrupt heard] {text}")
            self._cancel_active_reply_locked()
            interrupt_speech()

            # "stop" / "stop the music" jne. — suoritetaan music_stop aina kun kyseessä
            if local_command and local_command.get("action") == "music_stop":
                tool_result = handle_tool_speech(text, language=language)
                if tool_result.handled and tool_result.response:
                    self._speak_reply(tool_result.response)
                self._last_activity = now
                return True

            if not remainder:
                self._last_activity = now
                return True

            if is_echo:
                self._last_activity = now
                return True

            self._process_online_text(remainder, now=now, language=language, interrupted=True)
            return True

    # ------------------------------------------------------------------
    def bind_person(self, person_id: str | None) -> None:
        """Attach a recognized speaker to the next or current session."""
        with self._lock:
            self._person_id = person_id
            if person_id:
                self._store.touch_person(person_id)
                if self._session_id:
                    self._store.attach_person_to_session(self._session_id, person_id)

    def _start_session(self, now: float, wake_word: str, *, announce: bool) -> None:
        self._brain.reset()
        self._online = True
        self._last_activity = now
        self._session_id = self._store.start_session(person_id=self._person_id, wake_word=wake_word)
        self._store.add_event(
            "session_started",
            session_id=self._session_id,
            person_id=self._person_id,
            payload={"wake_word": wake_word},
        )
        print("[Engine] Wake word detected — going ONLINE.")
        ready_event = threading.Event()
        self._startup_vision_ready = ready_event
        threading.Thread(target=self._run_startup_vision, args=(ready_event,), daemon=True).start()
        if announce:
            self._speak_reply("Hei!")

    def _run_startup_vision(self, ready_event: threading.Event) -> None:
        """Capture a frame at session start to identify who's in the room."""
        import os
        if os.getenv("DISABLE_VISION", "").strip().lower() in ("1", "true", "yes", "on"):
            ready_event.set()
            return
        try:
            from vision.camera import Camera
            from vision.identity_manager import FaceManager
            with Camera(warmup_seconds=0.5) as cam:
                frame = cam.capture()
            names = FaceManager.get().recognize_faces(frame)
            if names:
                context = "Huoneessa tunnistettu: " + ", ".join(names) + "."
            else:
                context = "Kamerassa ei tunnistettuja henkilöitä."
            print(f"[Vision] Startup scan: {context}")
            self._brain.set_startup_context(context)
        except Exception as exc:
            print(f"[Vision] Startup scan failed: {exc}", file=sys.stderr)
        finally:
            ready_event.set()

    def _end_session(self, reason: str) -> None:
        with self._lock:
            self._cancel_active_reply_locked()
        session_id = self._session_id
        person_id = self._person_id
        if session_id:
            threading.Thread(
                target=self._brain.summarize_session,
                args=(session_id,),
                kwargs={"person_id": person_id},
                daemon=True,
            ).start()
            self._store.end_session(session_id, end_reason=reason)
            self._store.add_event(
                "session_ended",
                session_id=session_id,
                person_id=person_id,
                payload={"reason": reason},
            )
        self._online = False
        self._session_id = None
        self._session_language = ""
        self._brain.reset()
        print("[Engine] OFFLINE. Say 'founderbot', 'hei bot' or 'kuule bot' to wake me up.")

    def _process_online_text(
        self,
        text: str,
        *,
        now: float,
        language: str = "",
        interrupted: bool = False,
    ) -> None:
        self._last_activity = now
        # Kieli: Whisper → session muisti → tekstipohjainen arvio
        if language:
            self._session_language = language
        elif not self._session_language:
            inferred = _infer_language_from_text(text)
            if inferred:
                self._session_language = inferred
                language = inferred
        else:
            language = self._session_language
        # Hylätään kaiku: botti kuulee omansa mikistä
        refs = [r for r in (self._active_reply_text, self._last_spoken_text) if r and r.strip()]
        if refs and any(self._text_looks_like_echo(text, ref) for ref in refs):
            return
        print(f"[Online heard] {text} [lang={language or '?'}]")
        print(f"You said: {text}")
        self._log_user_message(text)

        if self._is_session_end_intent(text):
            # Sleep/lepotila: erillinen vastaus
            if self._is_sleep_intent(text):
                bye_msg = {"en": "Going to sleep. Wake me when you need me.", "fi": "Menen lepoon. Herätä kun tarvitset."}.get(language, "Menen lepoon.")
            else:
                bye_msg = {"en": "Okay, going offline.", "fi": "Hei hei!"}.get(language, "Hei hei!")
            self._speak_reply(bye_msg, end_session_reason="goodbye")
            return

        tool_result = handle_tool_speech(text, language=language)
        if tool_result.handled:
            if self._session_id and tool_result.action:
                self._store.log_tool_call(
                    tool_name=tool_result.action,
                    input_payload={"text": text},
                    output_summary=tool_result.response or "",
                    success=tool_result.success,
                    session_id=self._session_id,
                )
            if tool_result.response:
                print(f"[Tool] {tool_result.action}: {tool_result.response}")
                self._speak_reply(tool_result.response)
            return

        # No keyword match: LLM with tools (streaming so TTS can start early).
        self._start_streamed_reply_with_tools(
            text=text,
            session_id=self._session_id,
            person_id=self._person_id,
            language=language,
            interrupted=interrupted,
        )

    def _speak_reply(self, text: str, *, end_session_reason: str | None = None) -> SpeechHandle | None:
        text = (text or "").strip()
        if not text:
            if end_session_reason:
                self._end_session(end_session_reason)
            return None

        self._log_assistant_message(text)
        handle = speak(text)
        handle.add_done_callback(
            lambda finished_handle: self._on_reply_finished(
                finished_handle,
                end_session_reason=end_session_reason,
            )
        )
        return handle

    def _start_streamed_reply(
        self,
        *,
        text: str,
        session_id: str | None,
        person_id: str | None,
    ) -> None:
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_active_reply_locked()
            self._reply_generation += 1
            generation = self._reply_generation
            self._reply_cancel_event = cancel_event
            self._active_reply_text = ""
        threading.Thread(
            target=self._run_streamed_reply,
            args=(text, session_id, person_id, generation, cancel_event),
            daemon=True,
        ).start()

    def _start_streamed_reply_with_tools(
        self,
        *,
        text: str,
        session_id: str | None,
        person_id: str | None,
        language: str = "fi",
        interrupted: bool = False,
    ) -> None:
        """Stream LLM reply with tools; execute tool_calls after stream ends."""
        cancel_event = threading.Event()
        with self._lock:
            self._cancel_active_reply_locked()
            self._reply_generation += 1
            generation = self._reply_generation
            self._reply_cancel_event = cancel_event
            self._active_reply_text = ""
        threading.Thread(
            target=self._run_streamed_reply_with_tools,
            args=(text, session_id, person_id, generation, cancel_event),
            kwargs={"language": language, "interrupted": interrupted},
            daemon=True,
        ).start()

    def _run_streamed_reply_with_tools(
        self,
        text: str,
        session_id: str | None,
        person_id: str | None,
        generation: int,
        cancel_event: threading.Event,
        *,
        language: str = "",
        interrupted: bool = False,
    ) -> None:
        # Wait for startup vision (camera + face recognition) before the first LLM call
        # so the bot knows who's in the room when generating the greeting.
        ready_event = self._startup_vision_ready
        if ready_event is not None and not ready_event.is_set():
            ready_event.wait(timeout=1.5)
            self._startup_vision_ready = None  # Only gate the first turn

        tool_calls_out: list = []
        chunks = self._brain.stream_think_with_tools(
            text,
            session_id=session_id,
            person_id=person_id,
            stop_event=cancel_event,
            tool_calls_out=tool_calls_out,
            language=language,
            interrupted=interrupted,
        )
        full_reply = self._speak_streamed_reply(
            chunks,
            generation=generation,
            cancel_event=cancel_event,
        )
        if full_reply:
            print(f"[LLM] {full_reply}")

        # Tools whose result must always be spoken aloud (they ARE the answer).
        _ALWAYS_SPEAK = {"see", "lookup_knowledge", "get_menu"}

        tool_results: list[str] = []
        always_speak_results: list[str] = []
        for tc in tool_calls_out:
            name = getattr(getattr(tc, "function", None), "name", None) or ""
            raw_args = getattr(getattr(tc, "function", None), "arguments", None) or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            result = self._execute_llm_tool(name, args, language=language)
            if result:
                if name in _ALWAYS_SPEAK:
                    always_speak_results.append(result)
                else:
                    tool_results.append(result)
            if session_id:
                self._store.log_tool_call(
                    tool_name=name,
                    input_payload={"text": text, "arguments": args},
                    output_summary=result or "",
                    success=bool(result and "couldn't" not in result.lower()),
                    session_id=session_id,
                )

        # Always-speak tools: spoken regardless of whether LLM also produced text.
        for result in always_speak_results:
            self._speak_reply(result)

        # Other tools: speak only if LLM said nothing (tool result acts as the reply).
        if not full_reply and not always_speak_results and tool_results:
            reply = tool_results[0] if len(tool_results) == 1 else ". ".join(tool_results)
            self._speak_reply(reply)

        with self._lock:
            if self._reply_cancel_event is cancel_event:
                self._reply_cancel_event = None
            if generation == self._reply_generation and cancel_event.is_set():
                self._active_reply_text = ""

    def _run_streamed_reply(
        self,
        text: str,
        session_id: str | None,
        person_id: str | None,
        generation: int,
        cancel_event: threading.Event,
    ) -> None:
        reply = self._speak_streamed_reply(
            self._brain.stream_think(
                text,
                session_id=session_id,
                person_id=person_id,
                stop_event=cancel_event,
            ),
            generation=generation,
            cancel_event=cancel_event,
        )
        if reply:
            print(f"[LLM] {reply}")
        with self._lock:
            if self._reply_cancel_event is cancel_event:
                self._reply_cancel_event = None
            if generation == self._reply_generation and cancel_event.is_set():
                self._active_reply_text = ""

    # Pienempi kynnys = vähemmän TTS-kutsuja, lyhyet lauseet yhdistetään.
    _TTS_BATCH_MIN_CHARS = 45

    def _speak_streamed_reply(self, chunks, *, generation: int, cancel_event: threading.Event) -> str:
        last_handle: SpeechHandle | None = None
        reply_parts: list[str] = []
        buffer = ""
        for chunk in chunks:
            with self._lock:
                if cancel_event.is_set() or generation != self._reply_generation:
                    break
            cleaned = chunk.strip()
            if not cleaned:
                continue
            reply_parts.append(cleaned)
            buffer = (buffer + " " + cleaned).strip() if buffer else cleaned
            with self._lock:
                if cancel_event.is_set() or generation != self._reply_generation:
                    break
                self._active_reply_text = buffer
            if len(buffer) >= self._TTS_BATCH_MIN_CHARS:
                last_handle = speak(buffer)
                buffer = ""

        if buffer:
            last_handle = speak(buffer)
        full_reply = " ".join(reply_parts).strip()
        if not full_reply:
            return ""

        with self._lock:
            cancelled = cancel_event.is_set() or generation != self._reply_generation
        if cancelled:
            return full_reply

        self._log_assistant_message(full_reply)
        if last_handle is not None:
            last_handle.add_done_callback(
                lambda finished_handle: self._on_reply_finished(
                    finished_handle,
                    end_session_reason=None,
                )
            )
        return full_reply

    def _on_reply_finished(
        self,
        handle: SpeechHandle,
        *,
        end_session_reason: str | None,
    ) -> None:
        with self._lock:
            if not handle.interrupted.is_set():
                self._active_reply_text = ""
            self._last_activity = time.monotonic()
            if end_session_reason:
                self._end_session(end_session_reason)

    @staticmethod
    def _strip_phrase(text: str, phrase: str) -> str:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        stripped = pattern.sub("", text, count=1)
        return stripped.strip(" ,.!?:;-")

    def _is_session_end_intent(self, text: str) -> bool:
        normalized = self._normalize_intent_text(text)
        if not normalized:
            return False
        if "stop listening to" in normalized:
            return False
        return any(pattern.search(normalized) for pattern in SESSION_END_PATTERNS)

    def _is_sleep_intent(self, text: str) -> bool:
        """True jos käyttäjä haluaa botin mennä nukkumaan / sleep-tilaan."""
        normalized = self._normalize_intent_text(text)
        if not normalized:
            return False
        sleep_patterns = (
            r"\b(?:mene nukkumaan|mennä nukkumaan|nuku nyt|mene lepoon|lepotila|mene lepotilaan|siirry lepotilaan)\b",
            r"\b(?:go to sleep|sleep now|you can sleep|take a nap)\b",
        )
        return any(re.search(p, normalized, re.I) for p in sleep_patterns)

    @staticmethod
    def _normalize_intent_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    def _cancel_active_reply_locked(self) -> None:
        if self._reply_cancel_event is not None:
            self._reply_cancel_event.set()
            self._reply_cancel_event = None
        self._reply_generation += 1
        self._active_reply_text = ""

    def _looks_like_clear_interrupt(self, text: str) -> bool:
        """True vain jos puhe on selkeästi suunnattu botille (ei taustakeskustelu tai taustamelu).
        Kaksitasoinen: vahva adressointi → alhaisempi kynnys; heikko → korkeampi."""
        normalized = text.lower().strip()
        if not normalized:
            return False

        words = re.findall(r"\w+", normalized)
        addressing_count = sum(
            1 for kw in ADDRESSING_KEYWORDS
            if re.search(rf"\b{re.escape(kw)}\b", normalized)
        )
        has_wake = any(w in normalized for w in WAKE_WORDS)

        # Vahva adressointi: herätyssana tai 2+ adressointisanaa → keskeytys helpommin
        if has_wake or addressing_count >= 2:
            if len(words) >= 6 and len(normalized) >= 30:
                pass  # fall through to echo check
            else:
                return False
        # Heikko adressointi: 1 sana → korkeampi kynnys (taustakeskustelu ei keskeytä)
        elif addressing_count >= 1:
            if len(words) >= 10 and len(normalized) >= 50:
                pass
            else:
                return False
        else:
            return False

        # Ei kaiku: hylätään jos teksti muistuttaa botin omaa puhetta
        for ref_text in (self._active_reply_text, self._last_spoken_text):
            ref_words = set(re.findall(r"\w+", (ref_text or "").lower()))
            if ref_words:
                current_words = set(words)
                overlap = len(current_words & ref_words) / max(1, len(current_words))
                if overlap >= 0.4:
                    return False

        return True

    def _looks_like_echo(self, text: str) -> bool:
        """True if text is likely the bot's own TTS picked up by the mic (ei käsitellä)."""
        return self._text_looks_like_echo(text, self._active_reply_text)

    def _text_looks_like_echo(self, text: str, reference: str) -> bool:
        """True if text is largely the same as reference (bot's own speech). Kynnys 0.4 = vähemmän oman äänen kuulemista."""
        rem = text.lower().strip()
        ref = reference.lower()
        if not rem or not ref:
            return False
        words_rem = set(re.findall(r"\w+", rem))
        words_ref = set(re.findall(r"\w+", ref))
        if not words_rem:
            return False
        overlap = len(words_rem & words_ref) / len(words_rem)
        if overlap >= 0.4:
            return True
        if len(rem) >= 6 and rem in ref:
            return True
        return False

    def _execute_llm_tool(self, name: str, args: dict, *, language: str = "") -> str:
        """Execute a tool invoked by the LLM. Returns a short phrase for TTS in user's language."""
        from Tools import _tr

        lang = "en" if language == "en" else "fi"

        if name == "play_music":
            from Tools.music import play_async, is_genre_like

            query = (args.get("query") or "music").strip() or "music"
            play_async(query)
            if is_genre_like(query):
                return "Soitetaan." if lang == "fi" else "Playing."
            return f"{_tr('play_ok', lang)}: {query}."
        if name == "music_skip":
            from Tools.music import skip

            ok = skip()
            return f"{_tr('skip_ok', lang)}." if ok else f"{_tr('nothing_playing', lang)}."
        if name == "music_pause":
            from Tools.music import pause

            ok = pause()
            return f"{_tr('pause_ok', lang)}." if ok else f"{_tr('nothing_playing', lang)}."
        if name == "music_resume":
            from Tools.music import resume

            ok = resume()
            return f"{_tr('resume_ok', lang)}." if ok else f"{_tr('nothing_playing', lang)}."
        if name == "music_stop":
            from Tools.music import stop

            ok = stop()
            return f"{_tr('stop_ok', lang)}." if ok else f"{_tr('nothing_playing', lang)}."
        if name == "music_add_to_queue":
            from Tools.music import add_to_queue

            query = (args.get("query") or "").strip()
            if query:
                add_to_queue(query)
                return f"{_tr('queue_ok', lang)}: {query}."
            return "What should I add?" if lang == "en" else "Mitä lisätään?"
        if name == "music_volume_up":
            from Tools.music import volume_up

            vol = volume_up()
            return f"Volume {vol}%." if lang == "en" else f"Ääni {vol}%."
        if name == "music_volume_down":
            from Tools.music import volume_down

            vol = volume_down()
            return f"Volume {vol}%." if lang == "en" else f"Ääni {vol}%."
        if name == "get_menu":
            from Tools.menu import get_all_menus, get_menu

            restaurant = (args.get("restaurant") or "").strip().lower()
            if restaurant:
                return get_menu(restaurant)
            return get_all_menus()
        if name == "lookup_knowledge":
            query = (args.get("query") or "").strip() or "TRES SFP Robolabs"
            hits = self._store.search_knowledge(query, limit=6)
            if not hits:
                return "No info on this topic." if lang == "en" else "Ei tietoa tästä aiheesta."
            return "\n\n".join(hits[:6])
        if name == "see":
            from vision.scene import capture_and_describe

            question = (args.get("question") or "Kuvaile lyhyesti mitä näet.").strip()
            print(f"[Vision] Capturing frame for question: {question}")
            return capture_and_describe(question, self._brain._client)
        return ""

    def _log_user_message(self, text: str) -> None:
        if self._session_id:
            self._store.add_message(self._session_id, "user", text)

    def _log_assistant_message(self, text: str) -> None:
        if self._session_id and text:
            self._store.add_message(self._session_id, "assistant", text)
        self._last_spoken_text = text or self._last_spoken_text
