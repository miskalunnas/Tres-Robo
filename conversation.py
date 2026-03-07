"""Conversation orchestrator — sits between the audio pipeline and the brain.

State machine:
  OFFLINE: listens for wake word, ignores everything else.
  ONLINE:  passes every utterance to the LLM and speaks the reply.
           Returns to OFFLINE on inactivity timeout or a goodbye phrase.
"""
import time

from brain import Brain
from memory import MemoryStore
from Tools import handle_speech as handle_tool_speech
from voice.tts import speak

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

    # ------------------------------------------------------------------
    def handle(self, text: str, now: float) -> None:
        """Called by main.py for every transcribed utterance."""
        # Check inactivity timeout before anything else.
        if self._online and (now - self._last_activity) >= INACTIVITY_TIMEOUT:
            print(
                f"[Engine] No speech for {INACTIVITY_TIMEOUT:.0f}s — going OFFLINE."
            )
            self._end_session("timeout")

        normalized = text.lower()
        matched_wake_word = next((word for word in WAKE_WORDS if word in normalized), None)

        if not self._online:
            print(f"[Offline heard] {text}")
            if matched_wake_word:
                self._start_session(now, matched_wake_word)
            return

        # --- ONLINE mode ---
        self._last_activity = now
        print(f"[Online heard] {text}")
        print(f"You said: {text}")
        self._log_user_message(text)

        if any(w in normalized for w in GOODBYE_WORDS):
            reply = "Okay, going offline."
            self._log_assistant_message(reply)
            speak(reply)
            self._end_session("goodbye")
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
                self._log_assistant_message(tool_result.response)
                speak(tool_result.response)
            return

        print("[Brain] Thinking...")
        reply = self._brain.think(text, session_id=self._session_id, person_id=self._person_id)
        print(f"[LLM] {reply}")
        self._log_assistant_message(reply)
        speak(reply)
        self._last_activity = time.monotonic()  # reset timer after bot finishes speaking

    # ------------------------------------------------------------------
    def bind_person(self, person_id: str | None) -> None:
        """Attach a recognized speaker to the next or current session."""
        self._person_id = person_id
        if person_id:
            self._store.touch_person(person_id)
            if self._session_id:
                self._store.attach_person_to_session(self._session_id, person_id)

    def _start_session(self, now: float, wake_word: str) -> None:
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
        speak("Hey! I'm listening.")

    def _end_session(self, reason: str) -> None:
        if self._session_id:
            self._brain.summarize_session(self._session_id, person_id=self._person_id)
            self._store.end_session(self._session_id, end_reason=reason)
            self._store.add_event(
                "session_ended",
                session_id=self._session_id,
                person_id=self._person_id,
                payload={"reason": reason},
            )
        self._online = False
        self._session_id = None
        self._brain.reset()
        print(f"[Engine] OFFLINE. Say '{WAKE_WORDS[0]}' to wake me up.")

    def _log_user_message(self, text: str) -> None:
        if self._session_id:
            self._store.add_message(self._session_id, "user", text)

    def _log_assistant_message(self, text: str) -> None:
        if self._session_id and text:
            self._store.add_message(self._session_id, "assistant", text)
