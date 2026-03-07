"""Conversation orchestrator — sits between the audio pipeline and the brain.

State machine:
  OFFLINE: listens for wake word, ignores everything else.
  ONLINE:  passes every utterance to the LLM and speaks the reply.
           Returns to OFFLINE on inactivity timeout or a goodbye phrase.
"""
import time

from brain import Brain
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
INACTIVITY_TIMEOUT = 30.0  # seconds of silence before going offline


class ConversationEngine:
    wake_word = WAKE_WORDS[0]

    def __init__(self) -> None:
        self._brain = Brain()
        self._online = False
        self._last_activity = time.monotonic()

    # ------------------------------------------------------------------
    def handle(self, text: str, now: float) -> None:
        """Called by main.py for every transcribed utterance."""
        # Check inactivity timeout before anything else.
        if self._online and (now - self._last_activity) >= INACTIVITY_TIMEOUT:
            print(
                f"[Engine] No speech for {INACTIVITY_TIMEOUT:.0f}s — going OFFLINE."
            )
            self._end_session()

        normalized = text.lower()

        if not self._online:
            print(f"[Offline heard] {text}")
            if any(w in normalized for w in WAKE_WORDS):
                self._start_session(now)
            return

        # --- ONLINE mode ---
        self._last_activity = now
        print(f"[Online heard] {text}")
        print(f"You said: {text}")

        if any(w in normalized for w in GOODBYE_WORDS):
            reply = self._brain.think(text)
            speak(reply)
            self._end_session()
            return

        print("[Brain] Thinking...")
        reply = self._brain.think(text)
        print(f"[LLM] {reply}")
        speak(reply)

    # ------------------------------------------------------------------
    def _start_session(self, now: float) -> None:
        self._brain.reset()
        self._online = True
        self._last_activity = now
        print("[Engine] Wake word detected — going ONLINE.")
        speak("Hey! I'm listening.")

    def _end_session(self) -> None:
        self._online = False
        self._brain.reset()
        print(f"[Engine] OFFLINE. Say '{WAKE_WORDS[0]}' to wake me up.")
