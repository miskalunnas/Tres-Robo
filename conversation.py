"""Conversation orchestrator — sits between the audio pipeline and the brain.

State machine:
  OFFLINE: listens for wake word, ignores everything else.
  ONLINE:  passes every utterance to the LLM and speaks the reply.
           Returns to OFFLINE on inactivity timeout or a goodbye phrase.
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
    # Intended wake word + common Whisper mis-transcriptions
    "founderbot",
    "founderbott",
    "founder bot",
    "found a bot",
    "founder bott",
    # Finnish alternatives
    "hei botti",
    "founderbotti",
    "hei robotti",
]
GOODBYE_WORDS = ["goodbye", "bye", "näkemiin", "hei hei", "stop listening"]
INTERRUPT_WORDS = [
    "stop",
    "stop talking",
    "be quiet",
    "quiet",
    "pause",
    "wait",
    "hold on",
    "shut up",
]
INACTIVITY_TIMEOUT = 60.0  # seconds of silence before going offline


class ConversationEngine:
    wake_word = WAKE_WORDS[0]

    def __init__(self) -> None:
        self._store = MemoryStore()
        self._brain = Brain(self._store)
        self._online = False
        self._last_activity = time.monotonic()
        self._session_id: str | None = None
        self._person_id: str | None = None
        self._lock = threading.RLock()
        self._reply_cancel_event: threading.Event | None = None
        self._reply_generation = 0
        self._active_reply_text = ""

    # ------------------------------------------------------------------
    def handle(self, text: str, now: float) -> None:
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
                self._start_session(now, matched_wake_word, announce=not bool(remainder))
                if remainder:
                    self._process_online_text(remainder, now=now)
                return

            self._process_online_text(text, now=now)

    def handle_interruption(self, text: str, now: float) -> bool:
        """Handle an utterance captured while the bot is speaking."""
        with self._lock:
            normalized = text.lower()
            matched_wake_word = next((word for word in WAKE_WORDS if word in normalized), None)
            matched_interrupt = next((word for word in INTERRUPT_WORDS if word in normalized), None)
            local_command = parse_command(text)
            clear_interrupt = self._looks_like_clear_interrupt(text)
            if (
                not matched_wake_word
                and not matched_interrupt
                and local_command is None
                and not clear_interrupt
            ):
                return False

            print(f"[Interrupt heard] {text}")
            self._cancel_active_reply_locked()
            interrupt_speech()

            if matched_wake_word:
                remainder = self._strip_phrase(text, matched_wake_word)
            elif matched_interrupt:
                remainder = self._strip_phrase(text, matched_interrupt)
            else:
                remainder = text.strip()

            if not remainder:
                self._last_activity = now
                return True

            self._process_online_text(remainder, now=now)
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
        print(f"[Engine] OFFLINE. Say '{WAKE_WORDS[0]}' to wake me up.")

    def _process_online_text(self, text: str, *, now: float) -> None:
        self._last_activity = now
        print(f"[Online heard] {text}")
        print(f"You said: {text}")
        self._log_user_message(text)

        normalized = text.lower()
        if any(w in normalized for w in GOODBYE_WORDS):
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
            daemon=True,
        ).start()

    def _run_streamed_reply_with_tools(
        self,
        text: str,
        session_id: str | None,
        person_id: str | None,
        generation: int,
        cancel_event: threading.Event,
    ) -> None:
        tool_calls_out: list = []
        chunks = self._brain.stream_think_with_tools(
            text,
            session_id=session_id,
            person_id=person_id,
            stop_event=cancel_event,
            tool_calls_out=tool_calls_out,
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

    def _speak_streamed_reply(self, chunks, *, generation: int, cancel_event: threading.Event) -> str:
        last_handle: SpeechHandle | None = None
        reply_parts: list[str] = []
        for chunk in chunks:
            with self._lock:
                if cancel_event.is_set() or generation != self._reply_generation:
                    break
            cleaned = chunk.strip()
            if not cleaned:
                continue
            reply_parts.append(cleaned)
            with self._lock:
                if cancel_event.is_set() or generation != self._reply_generation:
                    break
                self._active_reply_text = " ".join(reply_parts).strip()
            last_handle = speak(cleaned)

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

    def _cancel_active_reply_locked(self) -> None:
        if self._reply_cancel_event is not None:
            self._reply_cancel_event.set()
            self._reply_cancel_event = None
        self._reply_generation += 1
        self._active_reply_text = ""

    def _looks_like_clear_interrupt(self, text: str) -> bool:
        normalized = text.lower().strip()
        if not normalized:
            return False

        words = re.findall(r"\w+", normalized)
        if len(words) < 2 and len(normalized) < 8:
            return False

        active_words = set(re.findall(r"\w+", self._active_reply_text.lower()))
        if active_words:
            current_words = set(words)
            overlap = len(current_words & active_words) / max(1, len(current_words))
            if overlap >= 0.8:
                return False

        return True

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
        return ""

    def _log_user_message(self, text: str) -> None:
        if self._session_id:
            self._store.add_message(self._session_id, "user", text)

    def _log_assistant_message(self, text: str) -> None:
        if self._session_id and text:
            self._store.add_message(self._session_id, "assistant", text)
