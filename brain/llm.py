"""Brain: calls Kimi K2 on Moonshot AI."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("MOONSHOT_MODEL", "kimi-k2")

SYSTEM_PROMPT = (
    "You are a friendly and curious robot assistant running on a Raspberry Pi 5 at the Tampere Entrepreneurship Society Club house. "
    "The user speaks to you and you reply out loud via a speaker, so keep your answers "
    "short, natural, and conversational — avoid bullet points, markdown, or long lists."
)


class Brain:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=os.environ["MOONSHOT_API_KEY"],
            base_url="https://api.moonshot.ai/v1",
        )
        self._history: list[dict] = []
        self._reset_history()
        print(f"[Brain] Using model: {MODEL}")

    def think(self, user_text: str) -> str:
        """Process a user utterance and return the robot's spoken reply."""
        self._history.append({"role": "user", "content": user_text})
        print(f"[Brain] Sending to API (model={MODEL}, history_len={len(self._history)})...")

        try:
            response = self._client.chat.completions.create(
                model=MODEL,
                messages=self._history,
                timeout=30,
            )
        except Exception as exc:
            print(f"[Brain] API error: {exc}")
            return "Sorry, I couldn't reach my brain right now."

        print("[Brain] Got response.")
        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history. Call when a session ends."""
        self._reset_history()
        print("[Brain] Conversation reset.")

    def _reset_history(self) -> None:
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]
