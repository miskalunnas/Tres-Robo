"""Conversation orchestrator — sits between the audio pipeline and the brain.

State machine:
  OFFLINE: listens for wake word, ignores everything else.
  ONLINE:  passes every utterance to the LLM and speaks the reply.
           Returns to OFFLINE on inactivity timeout or a clear end-of-chat intent.
"""
import json
import re
import threading
import time

from brain import Brain
from memory import MemoryStore
from Tools import handle_speech as handle_tool_speech
from Tools.commands import parse_command
from voice.tts import SpeechHandle, interrupt as interrupt_speech, speak

WAKE_WORDS = [
    # Founderbot (täsmälliset)
    "founderbot",
    "founderbott",
    "founder bot",
    "found a bot",
    "founder bott",
    "founderbotti",
    # Hei + bot
    "hei botti",
    "hei bot",
    "hei robot",
    "hei robotti",
    # Kuule + bot (selkeä adressointi)
    "kuule botti",
    "kuule bot",
    "bot kuule",
    "botti kuule",
    # Tervehdys + bot
    "terve botti",
    "terve bot",
    "moro botti",
    "moro bot",
    "moi botti",
    "moi bot",
    # Lyhyet (kauempaa puhuttaessa)
    "ok bot",
    "okay bot",
    "yo bot",
]
SESSION_END_PATTERNS = (
    re.compile(r"\b(?:goodbye|bye(?: bye)?|näkemiin|hei hei|moi moi)\b"),
    re.compile(
        r"\b(?:see you(?: later)?|talk to you later|catch you later|talk later)\b"
    ),
    re.compile(
        r"\b(?:go offline|go idle|go to sleep|sleep now|stop listening|you can stop listening|you can sleep|you can go to sleep)\b"
    ),
    re.compile(
        r"\b(?:mene lepotilaan|siirry lepotilaan|mene nukkumaan|voit mennä nukkumaan|voit mennä lepotilaan)\b"
    ),
    re.compile(
        r"\b(?:that's all(?: for now)?|that is all(?: for now)?|that's everything|that is everything|we(?: are|'re) done|i(?: am|'m) done(?: for now)?|done for now|nothing else|no more questions|all good now)\b"
    ),
    re.compile(
        r"\b(?:siinä kaikki|tässä kaikki|ei muuta|ei muuta tällä erää|ollaan valmiita|se oli siinä|palataan myöhemmin|jutellaan myöhemmin|puhutaan myöhemmin|jatketaan myöhemmin)\b"
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

INACTIVITY_TIMEOUT = 60.0  # seconds of silence before going offline


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
        self._reply_generation = 0
        self._active_reply_text = ""
        self._last_spoken_text = ""

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
            matched_wake_word = next((word for word in WAKE_WORDS if word in normalized), None)

            if not self._online:
                print(f"[Offline heard] {text}")
                if not matched_wake_word:
                    return
                remainder = self._strip_phrase(text, matched_wake_word)
                has_followup = bool(remainder.strip())
                # Jos vain herätyssana: ei "Hey I'm listening", vaan LLM vastaa persoonalla (mitä isäntä)
                text_to_process = remainder.strip() if has_followup else "mitä isäntä"
                self._start_session(now, matched_wake_word, announce=False)
                self._process_online_text(text_to_process, now=now, language=language)
                return

            self._process_online_text(text, now=now, language=language)

    def handle_interruption(self, text: str, now: float, *, language: str = "") -> bool:
        """Handle an utterance captured while the bot is speaking.
        Botti kuuntelee koko ajan; text = alusta lähtien kuunneltu puhe (koko segmentti).
        Päätetään reagoidaanko: vain selkeä keskeytys (wake word, stop, pitkä tarkoituksellinen lause)."""
        with self._lock:
            normalized = text.lower()
            matched_wake_word = next((word for word in WAKE_WORDS if word in normalized), None)
            matched_interrupt = next((word for word in INTERRUPT_WORDS if word in normalized), None)
            matched_session_end = self._is_session_end_intent(text)
            local_command = parse_command(text)
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
        if announce:
            self._speak_reply("Hey! I'm listening.")

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
        if self._last_spoken_text and self._text_looks_like_echo(text, self._last_spoken_text):
            self._last_spoken_text = ""
            return
        print(f"[Online heard] {text}")
        print(f"You said: {text}")
        self._log_user_message(text)

        if self._is_session_end_intent(text):
            self._speak_reply("Okay, going offline.", end_session_reason="goodbye")
            return

        tool_result = handle_tool_speech(text)
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

        tool_results: list[str] = []
        for tc in tool_calls_out:
            name = getattr(getattr(tc, "function", None), "name", None) or ""
            raw_args = getattr(getattr(tc, "function", None), "arguments", None) or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
            result = self._execute_llm_tool(name, args)
            if result:
                tool_results.append(result)
            if session_id:
                self._store.log_tool_call(
                    tool_name=name,
                    input_payload={"text": text, "arguments": args},
                    output_summary=result or "",
                    success=bool(result and "couldn't" not in result.lower()),
                    session_id=session_id,
                )

        if not full_reply and tool_results:
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
        Suodattaa: taustamelu, lyhyet äänähdykset, ihmiset jotka puhuvat toisilleen."""
        normalized = text.lower().strip()
        if not normalized:
            return False

        words = re.findall(r"\w+", normalized)
        # 1. Pituus: ei taustamelua, ei lyhyitä äänähdyksiä
        if len(words) < 8 or len(normalized) < 40:
            return False

        # 2. Adressointi: puheen pitää viitata botiin tai komentoihin — ei "mitä sanoit kokouksesta"
        if not any(
            re.search(rf"\b{re.escape(kw)}\b", normalized)
            for kw in ADDRESSING_KEYWORDS
        ):
            return False

        # 3. Ei kaiku: hylätään jos teksti muistuttaa botin omaa puhetta
        active_words = set(re.findall(r"\w+", self._active_reply_text.lower()))
        if active_words:
            current_words = set(words)
            overlap = len(current_words & active_words) / max(1, len(current_words))
            if overlap >= 0.5:
                return False

        return True

    def _looks_like_echo(self, text: str) -> bool:
        """True if text is likely the bot's own TTS picked up by the mic (ei käsitellä)."""
        return self._text_looks_like_echo(text, self._active_reply_text)

    def _text_looks_like_echo(self, text: str, reference: str) -> bool:
        """True if text is largely the same as reference (bot's own speech). Kynnys 0.5 = vähemmän vääriä keskeytyksiä."""
        rem = text.lower().strip()
        ref = reference.lower()
        if not rem or not ref:
            return False
        words_rem = set(re.findall(r"\w+", rem))
        words_ref = set(re.findall(r"\w+", ref))
        if not words_rem:
            return False
        overlap = len(words_rem & words_ref) / len(words_rem)
        if overlap >= 0.5:
            return True
        if len(rem) >= 8 and rem in ref:
            return True
        return False

    def _execute_llm_tool(self, name: str, args: dict) -> str:
        """Execute a tool invoked by the LLM. Returns a short phrase for TTS."""
        if name == "play_music":
            from Tools.music import play_async

            query = (args.get("query") or "music").strip() or "music"
            play_async(query)
            return f"Playing {query}."
        if name == "music_skip":
            from Tools.music import skip

            ok = skip()
            return "Skipping to next song." if ok else "Nothing playing to skip."
        if name == "music_pause":
            from Tools.music import pause

            ok = pause()
            return "Music paused." if ok else "Nothing to pause."
        if name == "music_resume":
            from Tools.music import resume

            ok = resume()
            return "Resuming playback." if ok else "Nothing to resume."
        if name == "music_stop":
            from Tools.music import stop

            ok = stop()
            return "Playback stopped." if ok else "Nothing was playing."
        if name == "music_add_to_queue":
            from Tools.music import add_to_queue

            query = (args.get("query") or "").strip()
            if query:
                add_to_queue(query)
                return f"Added {query} to the queue."
            return "What should I add to the queue?"
        if name == "music_volume_up":
            from Tools.music import volume_up

            vol = volume_up()
            return f"Volume {vol}%."
        if name == "music_volume_down":
            from Tools.music import volume_down

            vol = volume_down()
            return f"Volume {vol}%."
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
                return "No matching info in knowledge base."
            return "\n\n".join(hits[:6])
        return ""

    def _log_user_message(self, text: str) -> None:
        if self._session_id:
            self._store.add_message(self._session_id, "user", text)

    def _log_assistant_message(self, text: str) -> None:
        if self._session_id and text:
            self._store.add_message(self._session_id, "assistant", text)
        self._last_spoken_text = text or self._last_spoken_text
