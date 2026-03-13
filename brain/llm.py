"""Brain: calls GPT-4o-mini on OpenAI and enriches prompts with stored context."""

import json
import os
import re
import time
from collections.abc import Iterator
import threading
from types import SimpleNamespace

from dotenv import load_dotenv
from openai import OpenAI

from memory import MemoryStore

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DISABLE_VISION = os.getenv("DISABLE_VISION", "").strip().lower() in ("1", "true", "yes", "on")

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
with open(_PROMPT_FILE, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()

# Tools the LLM can call when it infers intent from conversation (e.g. play music).
LLM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "play_music",
            "description": (
                "Play music. Call when user asks to play music: 'play jazz', 'put on some chill', 'play lo-fi', 'soita jazz', 'laitetaan chill', 'taustamusiikkia'. "
                "query = YouTube search term. For genres use e.g. 'jazz music', 'chill music', 'lo-fi beats' — not just 'jazz'. "
                "For artists/songs: 'Beatles', 'Bohemian Rhapsody'. Call the tool — don't repeat back what the user said."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "YouTube search term: genre (jazz music, chill music, lo-fi), artist or song name.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_skip",
            "description": "Skip to next track. Use when: 'next', 'skip', 'next song', 'seuraava', 'vaihda biisi'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_pause",
            "description": "Pause music. Use when: 'pause', 'stop for a moment', 'tauko', 'pauseta'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_resume",
            "description": "Resume music. Use when: 'resume', 'continue', 'play again', 'jatka', 'jatka soitto'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_stop",
            "description": "Stop music and clear queue. Use when: 'stop music', 'turn off music', 'clear queue', 'lopeta', 'musiikki pois'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_add_to_queue",
            "description": "Add song to queue. Use when: 'add to queue', 'queue this', 'play next', 'lisää jonoon', 'laita seuraavaksi'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Song or artist name.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_volume_up",
            "description": "Volume up. Use when: 'louder', 'turn it up', 'volume up', 'kovemmalle', 'ääni ylös'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "music_volume_down",
            "description": "Volume down. Use when: 'quieter', 'turn it down', 'volume down', 'hiljemmalle', 'ääni alas'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_menu",
            "description": "Hae tämän päivän ruokalistat Hervannan kampukselta. Käytä kun käyttäjä kysyy ruokaa/lounasta. Vastaus on tiivis: päivä + yksi rivi per paikka (vain ruokalajit). Vastaa 1–2 lauseella, älä lue koko listaa ääneen. Ravintolat: reaktori, newton, konehuone, hertsi.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant": {
                        "type": "string",
                        "description": "Ravintola: reaktori, newton, konehuone, hertsi. Jätä tyhjäksi tai null = kaikki ruokalistat.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_knowledge",
            "description": (
                "Search the knowledge base for facts. CALL when: user asks about TRES, SFP, Robolabs, house people (Lauri, Netta, Olli...), slang (isäntä, raba, pöhinä), or bot persona. "
                "Use short query: 'TRES', 'SFP', 'robolabs', 'Lauri', 'bot_persona', 'isäntä'. Use returned info in your reply. Do not guess — fetch from DB."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, e.g. 'SFP dates', 'TRES benefits', 'robolabs', 'robot commands'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "see",
            "description": (
                "Your eyes — take a photo and answer a visual question. "
                "You MUST call this whenever the user asks about something that can only be answered by looking. "
                "TRIGGER: Call see if the user asks about: (1) how someone looks or feels (tired, happy, what they wear), "
                "(2) who is in the room or who is present, (3) what is visible (objects, room, colors, what's on the table), "
                "(4) body language or pose. "
                "Examples that REQUIRE see: 'mitä näet?', 'näytänkö väsyneeltä?', 'kuka siellä on?', 'what do you see?', "
                "'do I look happy?', 'what am I wearing?', 'onko täällä muita?', 'what's on the table?'. "
                "Never answer a visual question from memory or by guessing — always call see first. "
                "If the question is not about what is visible right now (e.g. time, music, menu), do NOT call see."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The exact visual question to answer from the photo, in the user's language. E.g. 'Does the person look tired?', 'Who is in the room?', 'What is the person wearing?'",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_knowledge",
            "description": (
                "Save a new fact to long-term memory. "
                "Call this silently whenever you hear something with lasting value: "
                "a new fact about a person in the house (job change, hobby, moved), "
                "a change to the space (new equipment, renovation, event), "
                "or something the user explicitly asks you to remember. "
                "Do NOT save: temporary states ('tired today', 'hungry'), chitchat, "
                "opinions without facts, things already in your knowledge, or uncertain info. "
                "After saving, continue the conversation naturally — never announce that you saved something "
                "or ask for permission first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": (
                            "The fact as a clear, standalone sentence with enough context to be useful later. "
                            "Good: 'Lauri got a new job at Siemens in early 2026.' "
                            "Good: 'There is a new espresso machine in the Robolabs kitchen.' "
                            "Bad: 'He is tired.' Bad: 'Interesting.'"
                        ),
                    }
                },
                "required": ["fact"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_conversation",
            "description": (
                "End the conversation and go offline. "
                "Call this when the exchange is clearly finished: user says thanks/bye/that's all, "
                "or the topic is fully resolved and the user shows no sign of continuing. "
                "Do NOT call for every short reply — only when the conversation is genuinely done. "
                "Provide a short, natural farewell in the user's language as the 'farewell' argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "farewell": {
                        "type": "string",
                        "description": "Short farewell in the user's language, e.g. 'Hei hei!', 'Moikka!', 'See you!', 'Catch you later!'",
                    }
                },
                "required": ["farewell"],
            },
        },
    },
]


def _get_llm_tools() -> list:
    """Tools for LLM; excludes see when DISABLE_VISION=1 (ei kameraa)."""
    tools = LLM_TOOLS
    if DISABLE_VISION:
        tools = [t for t in tools if t.get("function", {}).get("name") != "see"]
    return tools


class Brain:
    def __init__(self, store: MemoryStore | None = None) -> None:
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._store = store
        self._history: list[dict] = []
        self._startup_context: str = ""
        self._reset_history()
        print(f"[Brain] Using model: {MODEL}")
        if DISABLE_VISION:
            print("[Brain] Vision disabled (DISABLE_VISION=1, no camera)")

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
            return "En kuullut oikein."

        print(f"[Timing] LLM: {time.monotonic()-t0:.2f}s")
        reply = response.choices[0].message.content or ""
        if not session_id:
            self._history.append({"role": "assistant", "content": reply})
        return reply

    def think_with_tools(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        person_id: str | None = None,
    ) -> tuple[str, list]:
        """Call the LLM with tool definitions. Returns (content, tool_calls).
        tool_calls is a list of OpenAI tool_call objects (with .id, .function.name, .function.arguments).
        """
        messages = self._build_messages(user_text, session_id=session_id, person_id=person_id)
        print(f"[Brain] API call with tools (model={MODEL}, history_len={len(messages)})...")

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=_get_llm_tools(),
                tool_choice="auto",
                timeout=30,
            )
        except Exception as exc:
            print(f"[Brain] API error: {exc}")
            return "En kuullut oikein.", []

        print(f"[Timing] LLM with tools: {time.monotonic()-t0:.2f}s")
        msg = response.choices[0].message
        content = (msg.content or "").strip()
        tool_calls = getattr(msg, "tool_calls", None) or []

        if not session_id:
            self._history.append(
                {"role": "assistant", "content": content or "I used a tool."}
            )

        return content, list(tool_calls)

    def stream_think_with_tools(
        self,
        user_text: str,
        *,
        session_id: str | None = None,
        person_id: str | None = None,
        stop_event: threading.Event | None = None,
        tool_calls_out: list | None = None,
        language: str = "",
        interrupted: bool = False,
    ) -> Iterator[str]:
        """Stream a reply with tools enabled. Yields speakable content chunks.
        When the stream ends, appends parsed tool_calls to tool_calls_out (each item
        has .id, .function.name, .function.arguments for compatibility with OpenAI response).
        """
        if tool_calls_out is None:
            tool_calls_out = []
        messages = self._build_messages(
            user_text,
            session_id=session_id,
            person_id=person_id,
            language=language,
            interrupted=interrupted,
        )
        print(f"[Brain] Streaming API call with tools (model={MODEL}, history_len={len(messages)})...")

        t0 = time.monotonic()
        try:
            stream = self._client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=_get_llm_tools(),
                tool_choice="auto",
                timeout=30,
                stream=True,
            )
        except Exception as exc:
            print(f"[Brain] Streaming API error: {exc}")
            yield "En kuullut oikein."
            return

        buffer = ""
        emitted_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, str]] = {}
        try:
            for event in stream:
                if stop_event is not None and stop_event.is_set():
                    print("[Brain] Streaming cancelled.")
                    return
                if not event.choices:
                    continue
                delta = event.choices[0].delta

                if getattr(delta, "content", None):
                    buffer += delta.content
                    chunks, buffer = self._extract_speakable_chunks(buffer)
                    for chunk in chunks:
                        if stop_event is not None and stop_event.is_set():
                            return
                        emitted_parts.append(chunk)
                        yield chunk

                for tc in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tc, "index", tc.get("index") if isinstance(tc, dict) else 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    acc = tool_calls_acc[idx]
                    fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else {})
                    if isinstance(fn, dict):
                        acc["id"] += tc.get("id") or ""
                        acc["name"] += fn.get("name") or ""
                        acc["arguments"] += fn.get("arguments") or ""
                    else:
                        acc["id"] += getattr(tc, "id", None) or ""
                        acc["name"] += getattr(fn, "name", None) or ""
                        acc["arguments"] += getattr(fn, "arguments", None) or ""

        except Exception as exc:
            print(f"[Brain] Streaming interrupted: {exc}")
            if not emitted_parts:
                yield "No niin."
        finally:
            print(f"[Timing] LLM stream with tools: {time.monotonic()-t0:.2f}s")

        if stop_event is not None and stop_event.is_set():
            return
        tail = buffer.strip()
        if tail:
            emitted_parts.append(tail)
            yield tail

        # Build tool_calls list for the caller (same shape as OpenAI non-stream response).
        for idx in sorted(tool_calls_acc.keys()):
            acc = tool_calls_acc[idx]
            if acc["name"]:
                tool_calls_out.append(
                    SimpleNamespace(
                        id=acc["id"],
                        function=SimpleNamespace(
                            name=acc["name"],
                            arguments=acc["arguments"] or "{}",
                        ),
                    )
                )

        if not session_id and (emitted_parts or tool_calls_out):
            self._history.append(
                {"role": "assistant", "content": " ".join(emitted_parts).strip() if emitted_parts else "I used a tool."}
            )

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
            yield "En kuullut oikein."
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
                yield "No niin."
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

        transcript = self._store.render_session_transcript(session_id, limit=28)
        if not transcript.strip():
            return

        summary_prompt = (
            "Extract from this conversation:\n"
            "1) Up to 5 durable memory facts for a home robot assistant (stable facts: name preference, interests, routines, likes/dislikes).\n"
            "2) One short sentence summarizing what this session was about (e.g. 'Keskusteltiin musiikista ja tilattiin pizza.').\n"
            "Return strict JSON with exactly two keys: \"facts\" (list of objects with category, key, value, confidence) and \"session_summary\" (string, one sentence)."
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
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
        except Exception as exc:
            print(f"[Brain] Memory summary skipped: {exc}")
            return

        facts = data.get("facts") if isinstance(data, dict) else []
        session_summary = data.get("session_summary") if isinstance(data, dict) else None
        if isinstance(session_summary, str) and session_summary.strip():
            self._store.update_session_summary(session_id, session_summary.strip())

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

    def set_startup_context(self, text: str) -> None:
        """Inject context captured at session start (e.g. who the camera recognized)."""
        self._startup_context = text
        print(f"[Brain] Startup context: {text}")

    def reset(self) -> None:
        """Clear conversation history. Call when a session ends."""
        self._startup_context = ""
        self._reset_history()
        print("[Brain] Conversation reset.")

    def _build_messages(
        self,
        user_text: str,
        *,
        session_id: str | None,
        person_id: str | None,
        language: str = "",
        interrupted: bool = False,
    ) -> list[dict]:
        """Rakenna viestilista: system prompt + edellinen yhteenveto + 5 faktaa + (ehkä) 4 tietopohja-osumaa + 8 viimeisintä viestiä."""
        if self._store is None or not session_id:
            if language:
                _lh = {"fi": "suomea", "en": "englantia", "sv": "ruotsia", "de": "saksaa", "es": "espanjaa", "fr": "ranskaa"}
                ld = _lh.get(language, language)
                if language == "en":
                    lang_rule = "CRITICAL: The user is speaking English. Reply in natural, fluent English. Do not switch to Finnish or mix languages. Match their tone (casual/formal)."
                else:
                    lang_rule = f"KRIITTINEN: Käyttäjä puhuu {ld}. Vastaa luonnollisella kielellä. Älä vaihda toiselle kielelle tai sekoita kieliä."
                self._history.append({"role": "system", "content": lang_rule})
            self._history.append({"role": "user", "content": user_text})
            return self._history

        # KIELI ENAKKOON — botti vastaa aina käyttäjän kielellä
        messages: list[dict] = []
        if language:
            _lh = {"fi": "suomea", "en": "englantia", "sv": "ruotsia", "de": "saksaa", "es": "espanjaa", "fr": "ranskaa"}
            ld = _lh.get(language, language)
            if language == "en":
                lang_rule = "CRITICAL: The user is speaking English. Reply in natural, fluent English. Do not switch to Finnish or mix languages. Match their tone (casual/formal)."
            else:
                lang_rule = f"KRIITTINEN: Käyttäjä puhuu {ld}. Vastaa luonnollisella kielellä. Älä vaihda toiselle kielelle tai sekoita kieliä."
            messages.append({"role": "system", "content": lang_rule})
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
        if DISABLE_VISION:
            messages.append({
                "role": "system",
                "content": "Sinulla ei ole kameraa. Kun käyttäjä kysyy mitä näet, kuka siellä on tai vastaavaa, sano lyhyesti että sinulla ei ole silmiä tässä asennuksessa.",
            })
        if self._startup_context:
            messages.append({"role": "system", "content": f"Aloituskuva (kamera sessioalussa): {self._startup_context}"})
        prev_summary = self._store.get_previous_session_summary(
            person_id, exclude_session_id=session_id
        )
        if prev_summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"Edellinen istunto (yhteenveto): {prev_summary}",
                }
            )
        # Muisti aina; tietopohja aina (botti tarvitsee persoonan ja faktat)
        context_text = self._store.get_context_as_text(
            user_text,
            person_id=person_id,
            knowledge_limit=8,
            memory_limit=6,
            include_knowledge=True,
        )
        if context_text:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "KONTEKSTI (käytä tätä vastataksesi):\n"
                        "Muisti = käyttäjän/istunnon tiedot. Tietopohja = TRES, SFP, Robolabs, housen tyypit, slangit. "
                        "Jos vastaus on kontekstissa, käytä sitä. Jos puuttuu, kutsu lookup_knowledge.\n\n"
                        f"{context_text}"
                    ),
                }
            )
        # Keskeytys
        hints: list[str] = []
        if interrupted:
            hints.append("Käyttäjä keskeytti sinut juuri. Vastaa lyhyesti ja ota se huomioon.")
        if hints:
            messages.append(
                {"role": "system", "content": " ".join(hints)}
            )
        messages.append(
            {
                "role": "system",
                "content": (
                    "Vastaussäännöt: Viittaa edellisiin viesteihin ('se', 'tuo', 'sama'). Pysy samalla kielellä kuin käyttäjä. "
                    "TRES/SFP/Robolabs/housen tyypit/slangi: käytä kontekstin tietopohjaa. Jos vastaus puuttuu, kutsu lookup_knowledge(query='TRES') tms. Älä arvaa — hae tietokannasta. "
                    "Vastaa aina kontekstin perusteella."
                ),
            }
        )
        if language:
            _lang_label = "englanti" if language == "en" else {"fi": "suomi", "sv": "ruotsi"}.get(language, language)
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Istunnon kieli: englanti. Vastaa englanniksi."
                        if language == "en"
                        else f"Istunnon kieli: {_lang_label}. Vastaa samalla kielellä."
                    ),
                }
            )
        messages.extend(self._store.get_session_messages(session_id, limit=16))
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
