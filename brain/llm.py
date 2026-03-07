"""Brain: calls GPT-4o-mini on OpenAI and enriches prompts with stored context."""

import json
import os
import re
import time
from collections.abc import Iterator
import threading

from dotenv import load_dotenv
from openai import OpenAI

from memory import MemoryStore

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
with open(_PROMPT_FILE, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()


class Brain:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._store = store
        self._history: list[dict] = []
        self._reset_history()
        print(f"[Brain] Using model: {MODEL}")

    def think(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        person_id: str | None = None,
    ) -> str:
        """Process a user utterance and return the robot's spoken reply."""
        messages = self._build_messages(user_text, session_id=session_id, person_id=person_id)
        print(f"[Brain] Sending to API (model={MODEL}, history_len={len(messages)})...")

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                timeout=30,
            )
        except Exception as exc:
            print(f"[Brain] API error: {exc}")
            return "Sorry, I couldn't reach my brain right now."

        print(f"[Timing] LLM: {time.monotonic()-t0:.2f}s")
        reply = response.choices[0].message.content or ""
        if not session_id:
            self._history.append({"role": "assistant", "content": reply})
        return reply

    def stream_think(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        person_id: str | None = None,
        stop_event: threading.Event | None = None,
    ) -> Iterator[str]:
        """Yield a reply in speakable chunks as they arrive from the LLM."""
        messages = self._build_messages(user_text, session_id=session_id, person_id=person_id)
        print(f"[Brain] Streaming API call (model={MODEL}, history_len={len(messages)})...")

        t0 = time.monotonic()
        try:
            stream = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                timeout=30,
                stream=True,
            )
        except Exception as exc:
            print(f"[Brain] Streaming API error: {exc}")
            yield "Sorry, I couldn't reach my brain right now."
            return

        buffer = ""
        emitted_parts: list[str] = []
        try:
            for event in stream:
                if stop_event is not None and stop_event.is_set():
                    print("[Brain] Streaming cancelled.")
                    return
                delta = ""
                if event.choices:
                    delta = event.choices[0].delta.content or ""
                if not delta:
                    continue
                buffer += delta
                chunks, buffer = self._extract_speakable_chunks(buffer)
                for chunk in chunks:
                    if stop_event is not None and stop_event.is_set():
                        print("[Brain] Streaming cancelled.")
                        return
                    emitted_parts.append(chunk)
                    yield chunk
        except Exception as exc:
            print(f"[Brain] Streaming interrupted: {exc}")
            if not emitted_parts:
                yield "Sorry, I lost my train of thought."
            return
        finally:
            print(f"[Timing] LLM stream: {time.monotonic()-t0:.2f}s")

        if stop_event is not None and stop_event.is_set():
            print("[Brain] Streaming cancelled.")
            return
        tail = buffer.strip()
        if tail:
            emitted_parts.append(tail)
            yield tail

        if not session_id:
            self._history.append({"role": "assistant", "content": " ".join(emitted_parts).strip()})

    def summarize_session(self, session_id: str, *, person_id: str | None = None) -> None:
        """Extract durable facts from a finished session and store them."""
        if self._store is None:
            return

        transcript = self._store.render_session_transcript(session_id, limit=40)
        if not transcript.strip():
            return

        summary_prompt = (
            "Extract up to 5 durable memory facts for a home robot assistant. "
            "Only return stable, useful facts such as name preference, recurring interests, "
            "important routines, or explicit likes/dislikes. "
            "Do not include temporary requests or small talk. "
            "Return strict JSON as a list of objects with keys: category, key, value, confidence."
        )

        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": summary_prompt},
                    {"role": "user", "content": transcript},
                ],
                timeout=30,
            )
            raw = response.choices[0].message.content or "[]"
            facts = json.loads(raw)
        except Exception as exc:
            print(f"[Brain] Memory summary skipped: {exc}")
            return

        if not isinstance(facts, list):
            return

        for fact in facts[:5]:
            if not isinstance(fact, dict):
                continue
            category = str(fact.get("category", "")).strip() or "general"
            key = str(fact.get("key", "")).strip()
            value = str(fact.get("value", "")).strip()
            if not key or not value:
                continue
            confidence = fact.get("confidence", 0.5)
            try:
                confidence_value = max(0.0, min(1.0, float(confidence)))
            except (TypeError, ValueError):
                confidence_value = 0.5
            self._store.upsert_memory_fact(
                person_id=person_id,
                category=category,
                key=key,
                value=value,
                confidence=confidence_value,
            )

    def reset(self) -> None:
        """Clear conversation history. Call when a session ends."""
        self._reset_history()
        print("[Brain] Conversation reset.")

    def _build_messages(
        self,
        user_text: str,
        *,
        session_id: str | None,
        person_id: str | None,
    ) -> list[dict]:
        if self._store is None or not session_id:
            self._history.append({"role": "user", "content": user_text})
            return self._history

        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        memory_context = self._store.render_memory_context(person_id, limit=8)
        if memory_context:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Known durable context about the current user and prior sessions:\n"
                        f"{memory_context}"
                    ),
                }
            )
        messages.extend(self._store.get_session_messages(session_id, limit=20))
        return messages

    def _reset_history(self) -> None:
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]

    @staticmethod
    def _extract_speakable_chunks(buffer: str) -> tuple[list[str], str]:
        chunks: list[str] = []
        pattern = re.compile(r"(.+?[.!?]+(?:\s+|$))", re.DOTALL)
        while True:
            match = pattern.match(buffer)
            if match is None:
                break
            chunk = match.group(1).strip()
            if chunk:
                chunks.append(chunk)
            buffer = buffer[match.end():]
        return chunks, buffer
