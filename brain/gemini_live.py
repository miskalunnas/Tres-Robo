"""Gemini Live API session manager.

Replaces the Brain (GPT-4o-mini) + TTS (ElevenLabs) + STT (Whisper) pipeline
with a single persistent WebSocket connection to Gemini 2.5 Flash Live.

Architecture:
- Runs an asyncio event loop in a background daemon thread
- Thread-safe: send_audio() and close() can be called from any thread
- Callbacks are fired from the asyncio thread

Requirements:
    pip install google-genai
    GOOGLE_API_KEY= in .env
"""

import asyncio
import os
import sys
import threading
from collections.abc import Callable

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_LIVE_MODEL = os.environ.get("GEMINI_LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.environ.get("GEMINI_VOICE", "Puck")

# Input: 16kHz 16-bit PCM mono (matches webrtcvad / Whisper pipeline)
GEMINI_SAMPLE_RATE_IN = 16_000
# Output: Gemini sends 24kHz 16-bit PCM mono by default
GEMINI_SAMPLE_RATE_OUT = 24_000


class GeminiLiveSession:
    """Manages a single Gemini Live conversation session.

    Usage:
        session = GeminiLiveSession(
            system_prompt="...",
            tools=[...],             # OpenAI-style tool schemas
            tool_handler=fn,         # fn(name, args) -> str
            audio_out_handler=fn,    # fn(pcm_bytes: bytes) -> None
            on_session_end=fn,       # fn() -> None (called when session closes)
        )
        session.start()              # opens WebSocket in background thread
        session.send_audio(pcm)      # non-blocking, thread-safe
        session.close()              # end session gracefully
    """

    def __init__(
        self,
        system_prompt: str,
        tools: list,
        tool_handler: Callable[[str, dict], str],
        audio_out_handler: Callable[[bytes], None],
        on_session_end: Callable[[], None] | None = None,
        greeting: str = "Hei! Kuuntelen.",
    ) -> None:
        self._system_prompt = system_prompt
        self._tools = tools
        self._tool_handler = tool_handler
        self._audio_out_handler = audio_out_handler
        self._on_session_end = on_session_end
        self._greeting = greeting
        self._loop: asyncio.AbstractEventLoop | None = None
        self._audio_queue: asyncio.Queue | None = None
        self._text_queue: asyncio.Queue | None = None
        self._closed = False
        self._ready = threading.Event()
        self._send_count = 0

    def start(self) -> None:
        """Start the session in a background daemon thread."""
        t = threading.Thread(target=self._run_loop, daemon=True, name="gemini-live")
        t.start()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until the WebSocket session is open. Returns True if ready."""
        return self._ready.wait(timeout=timeout)

    def send_audio(self, pcm_bytes: bytes) -> None:
        """Thread-safe: enqueue raw 16kHz PCM bytes to send to Gemini."""
        if self._loop and self._audio_queue and not self._closed:
            self._loop.call_soon_threadsafe(self._safe_enqueue, self._audio_queue, pcm_bytes)
            self._send_count += 1
            if self._send_count in (1, 50, 200):
                print(f"[Gemini] Audio chunks sent: {self._send_count}")

    @staticmethod
    def _safe_enqueue(q: asyncio.Queue, item) -> None:
        """Drop oldest item if queue is full rather than raising QueueFull."""
        if q.full():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def send_text(self, text: str) -> None:
        """Thread-safe: send a text message to Gemini (end_of_turn=True)."""
        if self._loop and self._text_queue and not self._closed:
            self._loop.call_soon_threadsafe(self._text_queue.put_nowait, text)

    def close(self) -> None:
        """Thread-safe: close the session."""
        self._closed = True
        if self._loop and self._loop.is_running():
            # Unblock the send loop so it can exit cleanly
            if self._audio_queue:
                self._loop.call_soon_threadsafe(self._safe_enqueue, self._audio_queue, b"")
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------ internal

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._session_task())
        except Exception as exc:
            print(f"[Gemini] Session loop error: {exc}", file=sys.stderr)
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            if self._on_session_end:
                self._on_session_end()

    async def _session_task(self) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            print(
                "[Gemini] google-genai not installed.\n"
                "  Run: pip install google-genai",
                file=sys.stderr,
            )
            return

        if not GOOGLE_API_KEY:
            print("[Gemini] GOOGLE_API_KEY not set in .env", file=sys.stderr)
            return

        client = genai.Client(api_key=GOOGLE_API_KEY)
        self._audio_queue = asyncio.Queue(maxsize=512)
        self._text_queue = asyncio.Queue(maxsize=32)

        gemini_tools = _convert_tools_to_gemini(self._tools)
        use_tools = os.environ.get("GEMINI_TOOLS", "1").strip().lower() not in ("0", "false", "no")

        # Disable thinking (chain-of-thought) — reduces latency from ~10s to ~1s.
        # ThinkingConfig may not be available on all SDK versions; ignore if so.
        try:
            thinking_cfg = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            thinking_cfg = None

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=self._system_prompt)]
            ),
            tools=gemini_tools if (gemini_tools and use_tools) else None,
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=GEMINI_VOICE,
                    )
                ),
            ),
            **({} if thinking_cfg is None else {"thinking_config": thinking_cfg}),
        )

        print(f"[Gemini] Opening session (model={GEMINI_LIVE_MODEL}, voice={GEMINI_VOICE})")
        try:
            async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
                print("[Gemini] Session open — streaming audio.")
                self._ready.set()
                await asyncio.gather(
                    self._send_loop(session),
                    self._receive_loop(session),
                    self._text_loop(session),
                )
        except Exception as exc:
            if not self._closed:
                print(f"[Gemini] Session error: {exc}", file=sys.stderr)

    async def _send_loop(self, session) -> None:
        """Dequeue PCM frames, batch them, and stream to Gemini."""
        # 50ms batches — small enough to keep latency low, large enough to avoid
        # flooding the WebSocket with 20ms micro-frames.
        BATCH_INTERVAL = 0.05  # seconds
        buffer = bytearray()
        last_send = asyncio.get_event_loop().time()

        while not self._closed:
            try:
                pcm = await asyncio.wait_for(self._audio_queue.get(), timeout=0.05)
                buffer.extend(pcm)
            except asyncio.TimeoutError:
                pass
            except Exception as exc:
                if not self._closed:
                    print(f"[Gemini] Send error: {exc}", file=sys.stderr)
                break

            now = asyncio.get_event_loop().time()
            if buffer and (now - last_send) >= BATCH_INTERVAL:
                try:
                    await session.send(
                        input={"data": bytes(buffer), "mime_type": f"audio/pcm;rate={GEMINI_SAMPLE_RATE_IN}"},
                        end_of_turn=False,
                    )
                except Exception as exc:
                    if not self._closed:
                        print(f"[Gemini] Send error: {exc}", file=sys.stderr)
                    break
                buffer.clear()
                last_send = now

    async def _text_loop(self, session) -> None:
        """Send queued text messages to Gemini (used for follow-up turns)."""
        while not self._closed:
            try:
                text = await asyncio.wait_for(self._text_queue.get(), timeout=1.0)
                await session.send(input=text, end_of_turn=True)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:
                if not self._closed:
                    print(f"[Gemini] Text send error: {exc}", file=sys.stderr)
                break

    async def _receive_loop(self, session) -> None:
        """Receive audio output and tool calls from Gemini (multi-turn)."""
        audio_chunk_count = 0
        msg_count = 0
        try:
            while not self._closed:
                async for response in session.receive():
                    if self._closed:
                        break
                    msg_count += 1

                    # Extract audio: try response.data first, then inline_data
                    audio_data = getattr(response, "data", None)
                    if not audio_data:
                        sc = getattr(response, "server_content", None)
                        if sc:
                            model_turn = getattr(sc, "model_turn", None)
                            if model_turn:
                                for part in getattr(model_turn, "parts", []) or []:
                                    inline = getattr(part, "inline_data", None)
                                    if inline and getattr(inline, "data", None):
                                        audio_data = inline.data
                                        break

                    if audio_data:
                        audio_chunk_count += 1
                        if audio_chunk_count in (1, 10, 100):
                            print(f"[Gemini] Audio out chunk #{audio_chunk_count} ({len(audio_data)} bytes)")
                        self._audio_out_handler(audio_data)

                    # Tool calls
                    tool_call = getattr(response, "tool_call", None)
                    if tool_call:
                        await self._handle_tool_calls(session, tool_call)

        except Exception as exc:
            if not self._closed:
                print(f"[Gemini] Receive error: {exc}", file=sys.stderr)
        finally:
            print(f"[Gemini] Receive loop ended. Messages: {msg_count}, audio chunks: {audio_chunk_count}")

    async def _handle_tool_calls(self, session, tool_call) -> None:
        """Execute tool calls synchronously and send results back."""
        from google.genai import types

        responses = []
        for fc in tool_call.function_calls:
            name = fc.name
            args = dict(fc.args) if fc.args else {}
            print(f"[Gemini] Tool: {name}({args})")
            try:
                # Run blocking tool in thread pool so we don't block the event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda n=name, a=args: self._tool_handler(n, a)
                )
            except Exception as exc:
                result = f"Tool error: {exc}"
            responses.append(
                types.FunctionResponse(
                    id=fc.id,
                    name=name,
                    response={"result": result or ""},
                )
            )

        await session.send(
            input=types.LiveClientToolResponse(function_responses=responses)
        )


def _convert_tools_to_gemini(openai_tools: list) -> list:
    """Convert OpenAI-style tool list to a Gemini Tool list.

    OpenAI format:
        [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]

    Gemini format:
        [types.Tool(function_declarations=[types.FunctionDeclaration(...)])]
    """
    try:
        from google.genai import types
    except ImportError:
        return []

    _type_map = {
        "string": types.Type.STRING,
        "number": types.Type.NUMBER,
        "integer": types.Type.INTEGER,
        "boolean": types.Type.BOOLEAN,
        "object": types.Type.OBJECT,
        "array": types.Type.ARRAY,
    }

    declarations = []
    for tool in openai_tools:
        fn = tool.get("function", {})
        name = fn.get("name", "")
        if not name:
            continue
        description = fn.get("description", "")
        params = fn.get("parameters", {})

        properties = {}
        for prop_name, prop_schema in params.get("properties", {}).items():
            t = _type_map.get(prop_schema.get("type", "string"), types.Type.STRING)
            properties[prop_name] = types.Schema(
                type=t,
                description=prop_schema.get("description", ""),
            )

        if properties:
            schema = types.Schema(
                type=types.Type.OBJECT,
                properties=properties,
                required=params.get("required", []),
            )
        else:
            schema = types.Schema(type=types.Type.OBJECT)

        declarations.append(
            types.FunctionDeclaration(
                name=name,
                description=description,
                parameters=schema,
            )
        )

    if not declarations:
        return []
    return [types.Tool(function_declarations=declarations)]
