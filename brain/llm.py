"""Brain: calls Kimi K2 on Moonshot AI."""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("MOONSHOT_MODEL", "kimi-k2")

SYSTEM_PROMPT = (
    "You are Tres-Robo, a friendly and curious robot assistant running on a Raspberry Pi 5. "
    "The user speaks to you and you reply out loud via a speaker, so keep your answers "
    "short, natural, and conversational — avoid bullet points, markdown, or long lists."
)


class Brain:
    def __init__(self) -> None:
        self._client = OpenAI(
            api_key=os.environ["MOONSHOT_API_KEY"],
            base_url="https://api.moonshot.cn/v1",
        )
        self._history: list[dict] = []
        self._reset_history()
        print(f"[Brain] Using model: {MODEL}")

    def think(self, user_text: str) -> str:
        """Process a user utterance and return the robot's spoken reply."""
        self._history.append({"role": "user", "content": user_text})

        response = self._client.chat.completions.create(
            model=MODEL,
            messages=self._history,
        )

        reply = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history. Call when a session ends."""
        self._reset_history()
        print("[Brain] Conversation reset.")

    def _reset_history(self) -> None:
        self._history = [{"role": "system", "content": SYSTEM_PROMPT}]
