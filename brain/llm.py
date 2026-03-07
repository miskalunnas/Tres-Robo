"""Brain: calls GPT-4o-mini on OpenAI."""
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
with open(_PROMPT_FILE, encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()


class Brain:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
        )
        self._history: list[dict] = []
        self._reset_history()
        print(f"[Brain] Using model: {MODEL}")

    def think(self, user_text: str) -> str:
        """Process a user utterance and return the robot's spoken reply."""
        self._history.append({"role": "user", "content": user_text})
        print(f"[Brain] Sending to API (model={MODEL}, history_len={len(self._history)})...")

        t0 = time.monotonic()
        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=self._history,
                timeout=30,
            )
        except Exception as exc:
            print(f"[Brain] API error: {exc}")
            return "Sorry, I couldn't reach my brain right now."

        print(f"[Timing] LLM: {time.monotonic()-t0:.2f}s")
        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history. Call when a session ends."""
        self._reset_history()
        print("[Brain] Conversation reset.")

    def _reset_history(self) -> None:
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]
