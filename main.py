import queue
import sys
import time

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

try:
    import noisereduce as nr
except Exception:  # noqa: BLE001
    nr = None

from conversation import ConversationEngine

# WebRTC VAD only supports 8000, 16000, or 32000 Hz.
TARGET_SAMPLE_RATE = 16_000

# Voice activity detection settings.
VAD_FRAME_DURATION_MS = 30   # 10, 20 or 30 ms are valid
VAD_AGGRESSIVENESS = 2        # 0–3; higher = stricter noise rejection

# Segmenting: how much audio to collect before sending to Whisper.
MIN_SEGMENT_SECONDS = 0.5
MAX_SEGMENT_SECONDS = 8.0
MAX_SILENCE_BETWEEN_SPEECH_SECONDS = 0.8

# Set True to run noisereduce on each segment before Whisper (adds ~100 ms).
USE_DENOISER = False

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def float_to_int16_bytes(chunk: np.ndarray) -> bytes:
    """Convert float32 mono chunk to 16-bit PCM bytes for VAD."""
    if chunk.ndim > 1:
        chunk = chunk[:, 0]
    return np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def transcribe(model: WhisperModel, frames: list[np.ndarray]) -> str:
    """Concatenate frames, optionally denoise, then run Whisper."""
    audio = np.concatenate(frames, axis=0)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if USE_DENOISER and nr is not None:
        try:
            audio = nr.reduce_noise(y=audio, sr=TARGET_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            print(f"[Denoiser error] {exc}", file=sys.stderr)
    segments, info = model.transcribe(audio, task="transcribe")
    text = "".join(seg.text for seg in segments).strip()
    if text:
        print(f"[Whisper/{info.language}] {text}")
    return text


def listen_forever() -> None:
    try:
        device_info = sd.query_devices(kind="input")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Could not query audio devices: {exc}", file=sys.stderr)
        sys.exit(1)

    print(
        f"[Mic] {device_info['name']} — capturing at {TARGET_SAMPLE_RATE} Hz"
    )
    print("Loading Whisper model 'tiny'... (first run may take a moment)")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(TARGET_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)
    frame_seconds = VAD_FRAME_DURATION_MS / 1000.0

    engine = ConversationEngine()

    speech_frames: list[np.ndarray] = []
    segment_duration = 0.0
    silence_duration = 0.0

    with sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        blocksize=frame_length,
        device=None,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        try:
            while True:
                chunk = audio_queue.get()
                if chunk.size == 0:
                    continue

                is_speech = vad.is_speech(float_to_int16_bytes(chunk), TARGET_SAMPLE_RATE)

                if is_speech:
                    speech_frames.append(chunk)
                    segment_duration += frame_seconds
                    silence_duration = 0.0

                    if segment_duration >= MAX_SEGMENT_SECONDS:
                        text = transcribe(model, speech_frames)
                        speech_frames.clear()
                        segment_duration = 0.0
                        if text:
                            engine.handle(text, now=time.monotonic())

                else:
                    if speech_frames:
                        silence_duration += frame_seconds
                        if silence_duration >= MAX_SILENCE_BETWEEN_SPEECH_SECONDS:
                            if segment_duration >= MIN_SEGMENT_SECONDS:
                                text = transcribe(model, speech_frames)
                                if text:
                                    engine.handle(text, now=time.monotonic())
                            speech_frames.clear()
                            segment_duration = 0.0
                            silence_duration = 0.0

        except KeyboardInterrupt:
            print("\nStopping. Goodbye!")


if __name__ == "__main__":
    listen_forever()
