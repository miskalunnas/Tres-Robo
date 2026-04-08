"""Tres-Robo — Gemini Live architecture entry point.

Pipeline:
  OFFLINE  webrtcvad + Whisper (cheap, local wake word detection)
           → Wake word detected
  ONLINE   Raw audio frames streamed directly to Gemini Live WebSocket
           Gemini handles STT, LLM reasoning, TTS — all in one model
           Tools (music, menu, vision, etc.) executed locally on tool_call events
           → end_conversation tool OR inactivity timeout
  OFFLINE  WebSocket closed, back to wake word listening

Run with:
    GOOGLE_API_KEY=... python main_gemini.py

Requirements:
    pip install google-genai
"""

import os
import queue
import sys
import threading
import time

from dotenv import load_dotenv
load_dotenv()

import numpy as np

try:
    from face.display import FaceState, start_display, stop_display, set_state as face_set
    _face_enabled = True
except Exception as _face_err:
    _face_enabled = False
    def face_set(state): pass  # no-op when display unavailable
    def stop_display(): pass
    class FaceState:  # minimal stub
        IDLE = LISTENING = THINKING = SPEAKING = HAPPY = SAD = MUTED = None

# ── GPIO mute button ───────────────────────────────────────────────────────────
_MUTE_GPIO_PIN = int(os.environ.get("MUTE_GPIO_PIN", "17"))
_GPIO_DEBOUNCE_S = 0.25  # seconds

try:
    import lgpio  # type: ignore[import]
    _gpio_handle = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_input(_gpio_handle, _MUTE_GPIO_PIN, lgpio.SET_PULL_UP)
    _gpio_available = True
except Exception as _gpio_err:
    _gpio_available = False
    _gpio_handle = None
import requests
import sounddevice as sd
try:
    import webrtcvad  # type: ignore
except Exception:  # pragma: no cover
    webrtcvad = None

from brain.gemini_live import GeminiLiveSession, GEMINI_SAMPLE_RATE_IN, GEMINI_SAMPLE_RATE_OUT
from brain.llm import LLM_TOOLS, SYSTEM_PROMPT  # reuse existing tools + system prompt
from voice.audio_out import AudioPlayer
from memory.store import MemoryStore
from voice import stt_openai

# ── Wake word matching (inlined to avoid importing conversation.py's heavy deps) ─
import re

WAKE_WORDS = [
    "hei bot", "hei botti", "hei both", "hei robot", "hei robotti",
    "kuule bot", "kuule botti", "kuule both",
    "hey bot", "hey both", "hi bot", "hi both", "listen bot",
    "founderbot", "founder bot", "founder bott", "founderbott", "founderbotti",
    "found a bot", "founder",
    "bot", "robot", "robotti", "botti",
]


def _normalize_for_wake(text: str) -> str:
    s = text.lower().strip()
    s = re.sub(r"[,.!?:;]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _lenient_wake_match(text: str) -> str | None:
    n = _normalize_for_wake(text)
    words = set(re.findall(r"\b[a-zäö]+\b", n))
    if (words & {"hei", "hey", "hi", "kuule"}) and (words & {"bot", "botti", "both", "robot", "robotti"}):
        return "hei bot"
    return None

# ── Audio capture settings ─────────────────────────────────────────────────────
VAD_SAMPLE_RATE = 16_000
_vad_aggr = os.environ.get("VAD_AGGRESSIVENESS", "3").strip()
VAD_AGGRESSIVENESS = int(_vad_aggr) if _vad_aggr.isdigit() and 0 <= int(_vad_aggr) <= 3 else 2

_vad_frame_ms = os.environ.get("VAD_FRAME_DURATION_MS", "20").strip()
VAD_FRAME_DURATION_MS = int(_vad_frame_ms) if _vad_frame_ms.isdigit() and int(_vad_frame_ms) in (10, 20, 30) else 20

# Offline segmentation thresholds
MIN_SEGMENT_WHEN_OFFLINE = 0.25
MAX_SILENCE_WHEN_OFFLINE = 0.6
MAX_SEGMENT_OFFLINE = 8.0

# Session inactivity: close if Gemini hasn't produced audio for this long
INACTIVITY_TIMEOUT_SECONDS = 30.0

try:
    MIC_GAIN = float(os.environ.get("MIC_GAIN", "1.5").strip())
except (ValueError, TypeError):
    MIC_GAIN = 1.5
MIC_GAIN = max(0.5, min(5.0, MIC_GAIN))

WAKE_DEBUG = os.environ.get("WAKE_DEBUG", "").strip().lower() in ("1", "true", "yes")

WHISPER_PROMPT = (
    "Hei bot, hei botti, kuule bot, hey bot, hi bot, founderbot, founder bot. "
    "Hei, moi, hello, hi, hey. Lopetetaan, kiitos hei, bye, see you."
)

# ── Audio queue ────────────────────────────────────────────────────────────────
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)


def _put_latest(q: queue.Queue, item) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    data = np.clip(indata.copy() * MIC_GAIN, -1.0, 1.0).astype(np.float32)
    _put_latest(audio_queue, data)


# ── Audio helpers ──────────────────────────────────────────────────────────────

def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio
    target_len = int(len(audio) * to_sr / from_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def float_to_int16_bytes(chunk: np.ndarray) -> bytes:
    return np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def float32_to_pcm16(chunk: np.ndarray) -> bytes:
    """float32 mono array → 16-bit PCM bytes for Gemini input."""
    return np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16).tobytes()


# ── Tool execution ─────────────────────────────────────────────────────────────

_store = MemoryStore()
_openai_client = None  # lazy-init for vision

_PENDING_ACTION: dict | None = None


def _telegram_send_message_http(
    *,
    text: str,
    parse_mode: str | None = None,
    disable_web_page_preview: bool | None = None,
) -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    thread_id = os.environ.get("TELEGRAM_MESSAGE_THREAD_ID", "").strip()
    if not token:
        return "Telegram token puuttuu. Lisää .env:iin TELEGRAM_BOT_TOKEN."
    if not chat_id:
        return "Telegram chat id puuttuu. Lisää .env:iin TELEGRAM_CHAT_ID."

    payload: dict = {
        "chat_id": chat_id,
        "text": text,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if disable_web_page_preview is not None:
        payload["disable_web_page_preview"] = bool(disable_web_page_preview)
    if thread_id:
        # Optional: forum topic thread id in groups
        try:
            payload["message_thread_id"] = int(thread_id)
        except ValueError:
            return "TELEGRAM_MESSAGE_THREAD_ID on virheellinen (pitäisi olla numero)."

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    print(f"[Telegram] POST {url}")
    print(f"[Telegram] Payload: {payload}")
    try:
        r = requests.post(url, json=payload, timeout=10)
    except Exception as exc:
        print(f"[Telegram] Request failed: {exc}", file=sys.stderr)
        return f"Telegram-lähetys epäonnistui: {exc}"

    print(f"[Telegram] Response {r.status_code}: {r.text[:300]}")
    if r.status_code != 200:
        body = (r.text or "").strip()
        body = body[:500] + ("…" if len(body) > 500 else "")
        return f"Telegram API error ({r.status_code}): {body}"

    return "OK"


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _openai_client


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call from Gemini. Returns a result string.

    Gemini will speak this result in the user's language naturally —
    no need to localise here.
    """
    global _PENDING_ACTION
    if name == "play_music":
        from Tools.music import play_async, resolve_url, check_music_ready
        from Tools import _play_response_casual
        if not check_music_ready():
            return "Musiikki ei ole käytettävissä."
        query = (args.get("query") or "music").strip() or "music"
        url = resolve_url(query)
        if url is None:
            return "En löytänyt biisiä."
        play_async(query, url=url)
        return f"Playing: {query}"

    if name == "music_skip":
        from Tools.music import skip
        return "Seuraava." if skip() else "Ei ole mitään soimassa."

    if name == "music_pause":
        from Tools.music import pause
        return "Tauko." if pause() else "Ei ole mitään soimassa."

    if name == "music_resume":
        from Tools.music import resume
        return "Jatketaan." if resume() else "Ei ole mitään soimassa."

    if name == "music_stop":
        import subprocess as _sp
        from Tools.music import stop
        stop()
        for _cmd in ("mpv", "ffplay", "mpg123"):
            _sp.run(["pkill", "-f", _cmd], capture_output=True)
        return "Lopetettu."

    if name == "music_add_to_queue":
        from Tools.music import add_to_queue
        query = (args.get("query") or "").strip()
        if query:
            add_to_queue(query)
            return f"Lisätty jonoon: {query}."
        return "Mitä lisätään?"

    if name == "music_volume_up":
        from Tools.music import volume_up
        return f"Ääni {volume_up()}%."

    if name == "music_volume_down":
        from Tools.music import volume_down
        return f"Ääni {volume_down()}%."

    if name == "get_menu":
        from Tools.menu import get_all_menus, get_menu
        restaurant = (args.get("restaurant") or "").strip().lower()
        return get_menu(restaurant) if restaurant else get_all_menus()

    if name == "lookup_knowledge":
        query = (args.get("query") or "TRES SFP Robolabs").strip()
        hits = _store.search_knowledge(query, limit=6)
        return "\n\n".join(hits[:6]) if hits else "Ei tietoa tästä aiheesta."

    if name == "see":
        from vision.scene import capture_and_describe
        question = (args.get("question") or "Kuvaile lyhyesti mitä näet.").strip()
        print(f"[Vision] Capturing for: {question}")
        return capture_and_describe(question, _get_openai_client())

    if name == "save_knowledge":
        fact = (args.get("fact") or "").strip()
        if fact:
            _store.add_knowledge(source="conversation", content=fact)
            print(f"[Knowledge] Saved: {fact}")
            from memory.curator import curate_pending
            curate_pending()
        return "Saved."

    if name == "end_conversation":
        # Handled as a side effect by the caller; just return the farewell text
        return args.get("farewell") or "Hei hei!"

    if name == "telegram_send_message":
        text = (args.get("text") or "").strip()
        if not text:
            return "Mitä haluat lähettää Telegramiin?"

        parse_mode = (args.get("parse_mode") or "").strip() or None
        if parse_mode not in (None, "MarkdownV2", "Markdown", "HTML"):
            return "parse_mode pitää olla MarkdownV2, Markdown, HTML tai tyhjä."

        disable_preview = args.get("disable_web_page_preview", None)
        if disable_preview is not None:
            disable_preview = bool(disable_preview)

        _PENDING_ACTION = {
            "type": "telegram_send_message",
            "payload": {
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_preview,
            },
        }
        # Keep the tool response short — long strings going back to Gemini
        # can trigger 1011 server errors on the Live API.
        preview = text[:80] + ("…" if len(text) > 80 else "")
        return f"Viesti valmis: \"{preview}\". Lähetetäänkö? Kysy käyttäjältä kyllä/ei."

    if name == "confirm_action":
        decision = (args.get("decision") or "").strip().lower()
        if decision not in ("yes", "no"):
            return "Vastaa yes/no."
        if _PENDING_ACTION is None:
            print("[Telegram] confirm_action called but nothing was staged — telegram_send_message was not called first.", file=sys.stderr)
            return "Ei ole mitään vahvistettavaa."

        pending = _PENDING_ACTION
        _PENDING_ACTION = None  # clear first to avoid double-sends on retries

        if decision == "no":
            print("[Telegram] Send cancelled by user.")
            return "Selvä. Peruin lähetyksen."

        if pending.get("type") != "telegram_send_message":
            print(f"[Telegram] Unknown pending action type: {pending.get('type')}", file=sys.stderr)
            return "Vahvistus epäonnistui: tuntematon pending-toiminto."

        payload = pending.get("payload") or {}
        result = _telegram_send_message_http(
            text=str(payload.get("text") or ""),
            parse_mode=payload.get("parse_mode"),
            disable_web_page_preview=payload.get("disable_web_page_preview"),
        )
        if result == "OK":
            print("[Telegram] Message sent successfully.")
            return "Lähetetty Telegramiin."
        print(f"[Telegram] Send failed: {result}", file=sys.stderr)
        return f"En saanut lähetettyä Telegramiin: {result}"

    return f"Unknown tool: {name}"


# ── Startup face recognition ───────────────────────────────────────────────────

def _run_startup_vision() -> str:
    """Capture a frame and identify faces. Returns context string for system prompt."""
    try:
        from vision.camera import Camera
        from vision.identity_manager import FaceManager
        with Camera(warmup_seconds=0.5) as cam:
            frame = cam.capture()
        names = FaceManager.get().recognize_faces(frame)
        if names:
            return f"Aloituskuva (kamera sessioalussa): Huoneessa tunnistettu: {', '.join(names)}."
        return "Aloituskuva: Ei tunnistettuja henkilöitä huoneessa."
    except Exception as exc:
        print(f"[Vision] Startup scan skipped: {exc}", file=sys.stderr)
        return ""


# ── Whisper (offline wake word detection only) ─────────────────────────────────

def _transcribe_offline(frames: list[np.ndarray], native_sr: int) -> str:
    """Transcribe a short offline segment. Returns text only (language not needed)."""
    audio = np.concatenate(frames, axis=0)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = resample(audio, native_sr, VAD_SAMPLE_RATE)
    peak = np.abs(audio).max()
    if peak < 0.05:
        return ""
    if peak > 1e-4:
        audio = audio * min(0.95 / peak, 4.0)

    stt_lang = os.environ.get("STT_LANGUAGE", "").strip().lower()
    language = stt_lang if stt_lang in ("fi", "en", "sv") else None
    text, lang = stt_openai.transcribe(
        audio, VAD_SAMPLE_RATE,
        language=language,
        prompt=WHISPER_PROMPT,
        return_language=True,
    )
    if text:
        print(f"[Whisper/{lang or '?'}] {text}")
    return text


def _strip_wake_word(text: str, wake_word: str) -> str:
    normalized = text.lower()
    idx = normalized.find(wake_word)
    if idx >= 0:
        return text[idx + len(wake_word):].strip(" ,.-:!?")
    return text


# ── Device resolution ──────────────────────────────────────────────────────────

def _resolve_device_sample_rate(device, channels: int) -> int:
    _sr = os.environ.get("MIC_SAMPLE_RATE", "").strip()
    if _sr.isdigit():
        return int(_sr)
    dev = device if device is not None else sd.default.device[0]
    try:
        info = sd.query_devices(dev, "input")
        rate = int(info.get("default_samplerate", 0) or 0)
        if rate > 0:
            sd.check_input_settings(device=device, channels=channels, samplerate=rate)
            return rate
    except Exception:
        pass
    for rate in (16_000, 48_000, 44_100):
        try:
            sd.check_input_settings(device=device, channels=channels, samplerate=rate)
            return rate
        except sd.PortAudioError:
            continue
    return 48_000


# ── Main loop ──────────────────────────────────────────────────────────────────

def listen_forever() -> None:
    _default_dev = "" if sys.platform == "win32" else "hw:2,0"
    _dev = os.environ.get("MIC_DEVICE", _default_dev).strip()
    MIC_DEVICE = int(_dev) if _dev.isdigit() else (_dev if _dev else None)
    MIC_CHANNELS = int(os.environ.get("MIC_CHANNELS", "1"))

    native_sr = _resolve_device_sample_rate(MIC_DEVICE, MIC_CHANNELS)
    capture_block = int(native_sr * VAD_FRAME_DURATION_MS / 1000)
    frame_seconds = VAD_FRAME_DURATION_MS / 1000.0
    vad_frame_len = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

    if _face_enabled:
        start_display()
        face_set(FaceState.IDLE)

    # ── Mute state ─────────────────────────────────────────────────────────────
    muted = threading.Event()  # set = muted, clear = active

    def _on_button_press(channel):
        if muted.is_set():
            muted.clear()
            print("[Mute] Unmuted.")
            face_set(FaceState.IDLE)
        else:
            muted.set()
            print("[Mute] Muted.")
            if state["online"]:
                end_session("muted")
            face_set(FaceState.MUTED)

    if _gpio_available:
        _last_press = [0.0]

        def _gpio_poll():
            prev = 1  # pulled high = unpressed
            while True:
                try:
                    val = lgpio.gpio_read(_gpio_handle, _MUTE_GPIO_PIN)
                except Exception:
                    break
                if val == 0 and prev == 1:  # falling edge
                    now_t = time.monotonic()
                    if now_t - _last_press[0] >= _GPIO_DEBOUNCE_S:
                        _last_press[0] = now_t
                        _on_button_press(None)
                prev = val
                time.sleep(0.02)  # 20 ms poll interval

        threading.Thread(target=_gpio_poll, daemon=True, name="gpio-poll").start()
        print(f"[Mute] Button ready on GPIO {_MUTE_GPIO_PIN}.")
    else:
        print("[Mute] lgpio not available — mute button disabled.", file=sys.stderr)

    print(f"[Mic] device={MIC_DEVICE} channels={MIC_CHANNELS} {native_sr} Hz")
    print(f"[Gemini] Model: {os.environ.get('GEMINI_LIVE_MODEL', 'gemini-live-2.5-flash-native-audio')}")

    if webrtcvad is None:
        raise RuntimeError(
            "webrtcvad is not installed. Install dependencies from requirements.txt "
            "(on Windows, you may need Microsoft C++ Build Tools)."
        )
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    audio_player = AudioPlayer(sample_rate=GEMINI_SAMPLE_RATE_OUT)

    # Mutable state (using a dict so closures can modify)
    state = {
        "online": False,
        "session": None,            # GeminiLiveSession | None
        "last_audio_out": 0.0,      # updated when Gemini sends audio
        "last_audio_in": 0.0,       # updated while user is speaking
        "end_requested": False,     # set when end_conversation fires
        "session_closed_at": 0.0,   # for 409 cooldown between sessions
    }

    # Minimum seconds between session close and next open (avoids 409 Conflict)
    SESSION_COOLDOWN = 5.0

    # Offline VAD state
    speech_frames: list[np.ndarray] = []
    segment_duration = 0.0
    silence_duration = 0.0

    # ── callbacks ──────────────────────────────────────────────────────────────

    def on_audio_out(pcm_bytes: bytes) -> None:
        state["last_audio_out"] = time.monotonic()
        face_set(FaceState.SPEAKING)
        audio_player.play(pcm_bytes)

    def on_tool_call(name: str, args: dict) -> str:
        result = execute_tool(name, args)
        if name == "end_conversation":
            state["end_requested"] = True
        return result

    def on_session_end() -> None:
        print("[Gemini] Session ended → OFFLINE")
        face_set(FaceState.IDLE)
        state["online"] = False
        state["session"] = None
        state["session_closed_at"] = time.monotonic()

    def start_session() -> None:
        nonlocal segment_duration, silence_duration
        # Enforce cooldown to avoid 409 Conflict (server-side session still closing)
        since_close = time.monotonic() - state["session_closed_at"]
        if since_close < SESSION_COOLDOWN:
            wait = SESSION_COOLDOWN - since_close
            print(f"[Engine] Waiting {wait:.1f}s for session cooldown...")
            time.sleep(wait)
        print("[Engine] Wake word → ONLINE (Gemini Live)")
        audio_player.stop()
        speech_frames.clear()
        segment_duration = 0.0
        silence_duration = 0.0

        # Language override — native audio models need this prominent.
        lang_prefix = "CRITICAL: Always reply in the same language the user just spoke. English input → English reply. Finnish input → Finnish reply. Never switch language mid-response.\n\n"
        full_prompt = lang_prefix + SYSTEM_PROMPT

        session = GeminiLiveSession(
            system_prompt=full_prompt,
            tools=LLM_TOOLS,
            tool_handler=on_tool_call,
            audio_out_handler=on_audio_out,
            on_session_end=on_session_end,
        )
        session.start()
        if not session.wait_ready(timeout=10.0):
            print("[Engine] Gemini session failed to open in time.", file=sys.stderr)
            session.close()
            return
        state["session"] = session
        state["online"] = True
        state["last_audio_out"] = time.monotonic()
        state["last_audio_in"] = time.monotonic()
        state["end_requested"] = False
        face_set(FaceState.THINKING)

        # Run vision in background — inject context once camera finishes
        # so the session opens immediately without waiting for the camera.
        def _vision_and_inject():
            ctx = _run_startup_vision()
            if ctx and state["session"] is session:
                print(f"[Vision] {ctx}")
                session.send_text(f"[Context] {ctx}")
        threading.Thread(target=_vision_and_inject, daemon=True, name="startup-vision").start()

    def end_session(reason: str = "timeout") -> None:
        print(f"[Engine] → OFFLINE ({reason})")
        face_set(FaceState.IDLE)
        audio_player.stop()
        session = state["session"]
        if session:
            session.close()
        state["session"] = None
        state["online"] = False
        state["end_requested"] = False

    def check_wake(text: str) -> None:
        normalized = text.lower()
        normalized_flex = _normalize_for_wake(text)
        wake_word = next(
            (w for w in WAKE_WORDS if w in normalized or w in normalized_flex), None
        )
        if not wake_word:
            wake_word = _lenient_wake_match(text)
        if wake_word:
            start_session()
        elif WAKE_DEBUG:
            print(f"[Wake debug] No wake word in: {repr(text)}", file=sys.stderr)

    # ── audio loop ─────────────────────────────────────────────────────────────

    with sd.InputStream(
        samplerate=native_sr,
        blocksize=capture_block,
        device=MIC_DEVICE,
        dtype="float32",
        channels=MIC_CHANNELS,
        callback=audio_callback,
    ):
        print("[Engine] OFFLINE. Say 'hei bot', 'kuule bot' or 'founderbot' to wake up.")
        try:
            while True:
                now = time.monotonic()
                chunk = audio_queue.get()
                if chunk.size == 0:
                    continue

                # Mono conversion
                if chunk.ndim > 1:
                    mono = chunk.mean(axis=1) if MIC_CHANNELS > 1 else chunk[:, 0]
                else:
                    mono = chunk

                if muted.is_set():
                    continue  # drop all audio while muted

                if state["online"]:
                    # ── ONLINE: stream raw audio to Gemini ───────────────────
                    session = state["session"]
                    # Switch face to LISTENING once bot stops speaking
                    if not audio_player.recently_played(cooldown=0.5):
                        face_set(FaceState.LISTENING)
                    # Suppress mic while bot is speaking (queue busy) OR just
                    # finished (cooldown covers paplay's internal buffer drain).
                    if session and not audio_player.recently_played(cooldown=0.5):
                        pcm16 = float32_to_pcm16(
                            resample(mono, native_sr, GEMINI_SAMPLE_RATE_IN)
                        )
                        session.send_audio(pcm16)
                        state["last_audio_in"] = now  # reset while user is speaking

                    # Inactivity: time out if neither side has produced audio recently
                    last_activity = max(state["last_audio_out"], state["last_audio_in"])
                    if (now - last_activity) > INACTIVITY_TIMEOUT_SECONDS:
                        end_session("timeout")

                    # LLM-requested conversation end
                    if state["end_requested"]:
                        time.sleep(2.0)  # let Gemini finish speaking farewell
                        end_session("llm_goodbye")

                else:
                    # ── OFFLINE: webrtcvad + Whisper wake word detection ──────
                    mono_16k = resample(mono, native_sr, VAD_SAMPLE_RATE)
                    if len(mono_16k) > vad_frame_len:
                        mono_16k = mono_16k[:vad_frame_len]
                    elif len(mono_16k) < vad_frame_len:
                        mono_16k = np.pad(mono_16k, (0, vad_frame_len - len(mono_16k)))

                    is_speech = vad.is_speech(float_to_int16_bytes(mono_16k), VAD_SAMPLE_RATE)

                    if is_speech:
                        speech_frames.append(mono)
                        segment_duration += frame_seconds
                        silence_duration = 0.0

                        if segment_duration >= MAX_SEGMENT_OFFLINE:
                            text = _transcribe_offline(speech_frames, native_sr)
                            speech_frames.clear()
                            segment_duration = 0.0
                            if text:
                                check_wake(text)
                    else:
                        if speech_frames:
                            silence_duration += frame_seconds
                            if silence_duration >= MAX_SILENCE_WHEN_OFFLINE:
                                if segment_duration >= MIN_SEGMENT_WHEN_OFFLINE:
                                    text = _transcribe_offline(speech_frames, native_sr)
                                    if text:
                                        check_wake(text)
                                speech_frames.clear()
                                segment_duration = 0.0
                                silence_duration = 0.0

        except KeyboardInterrupt:
            print("\nStopping.")
            end_session("shutdown")
            audio_player.shutdown()
            stop_display()
            if _gpio_available:
                lgpio.gpiochip_close(_gpio_handle)
        except Exception:
            audio_player.shutdown()
            stop_display()
            if _gpio_available:
                lgpio.gpiochip_close(_gpio_handle)
            raise


if __name__ == "__main__":
    listen_forever()
