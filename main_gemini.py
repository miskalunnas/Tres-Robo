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
import sounddevice as sd
import webrtcvad

from brain.gemini_live import GeminiLiveSession, GEMINI_SAMPLE_RATE_IN, GEMINI_SAMPLE_RATE_OUT
from brain.llm import LLM_TOOLS, SYSTEM_PROMPT  # reuse existing tools + system prompt
from voice.audio_out import AudioPlayer
from conversation import WAKE_WORDS, _lenient_wake_match, _normalize_for_wake
from memory.store import MemoryStore
from voice import stt_openai

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
INACTIVITY_TIMEOUT_SECONDS = 40.0

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
        return _play_response_casual(query, "fi")

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
        from Tools.music import stop
        return "Lopetettu." if stop() else "Ei ole mitään soimassa."

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

    if name == "end_conversation":
        # Handled as a side effect by the caller; just return the farewell text
        return args.get("farewell") or "Hei hei!"

    return f"Unknown tool: {name}"


# ── Startup face recognition ───────────────────────────────────────────────────

def _run_startup_vision() -> str:
    """Capture a frame and identify faces. Returns context string for system prompt."""
    try:
        from vision.camera import Camera
        from vision.face_id import identify_faces
        with Camera(warmup_seconds=0.5) as cam:
            frame = cam.capture()
        names = identify_faces(frame)
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

    print(f"[Mic] device={MIC_DEVICE} channels={MIC_CHANNELS} {native_sr} Hz")
    print(f"[Gemini] Model: {os.environ.get('GEMINI_LIVE_MODEL', 'gemini-live-2.5-flash-native-audio')}")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    audio_player = AudioPlayer(sample_rate=GEMINI_SAMPLE_RATE_OUT)

    # Mutable state (using a dict so closures can modify)
    state = {
        "online": False,
        "session": None,            # GeminiLiveSession | None
        "last_audio_out": 0.0,      # for inactivity timeout
        "end_requested": False,     # set when end_conversation fires
    }

    # Offline VAD state
    speech_frames: list[np.ndarray] = []
    segment_duration = 0.0
    silence_duration = 0.0

    # ── callbacks ──────────────────────────────────────────────────────────────

    def on_audio_out(pcm_bytes: bytes) -> None:
        state["last_audio_out"] = time.monotonic()
        audio_player.play(pcm_bytes)

    def on_tool_call(name: str, args: dict) -> str:
        result = execute_tool(name, args)
        if name == "end_conversation":
            state["end_requested"] = True
        return result

    def on_session_end() -> None:
        print("[Gemini] Session ended → OFFLINE")
        state["online"] = False
        state["session"] = None

    def start_session() -> None:
        print("[Engine] Wake word → ONLINE (Gemini Live)")
        audio_player.stop()

        startup_ctx = _run_startup_vision()
        full_prompt = (startup_ctx + "\n\n" + SYSTEM_PROMPT) if startup_ctx else SYSTEM_PROMPT
        if startup_ctx:
            print(f"[Vision] {startup_ctx}")

        session = GeminiLiveSession(
            system_prompt=full_prompt,
            tools=LLM_TOOLS,
            tool_handler=on_tool_call,
            audio_out_handler=on_audio_out,
            on_session_end=on_session_end,
        )
        session.start()
        state["session"] = session
        state["online"] = True
        state["last_audio_out"] = time.monotonic()
        state["end_requested"] = False

    def end_session(reason: str = "timeout") -> None:
        print(f"[Engine] → OFFLINE ({reason})")
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

                if state["online"]:
                    # ── ONLINE: stream raw audio to Gemini ───────────────────
                    session = state["session"]
                    if session:
                        pcm16 = float32_to_pcm16(
                            resample(mono, native_sr, GEMINI_SAMPLE_RATE_IN)
                        )
                        session.send_audio(pcm16)

                    # Inactivity timeout
                    if (now - state["last_audio_out"]) > INACTIVITY_TIMEOUT_SECONDS:
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


if __name__ == "__main__":
    listen_forever()
