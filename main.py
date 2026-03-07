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

# Rate expected by webrtcvad and Whisper. Capture may differ — we resample.
VAD_SAMPLE_RATE = 16_000

# Voice activity detection settings.
VAD_FRAME_DURATION_MS = 30   # 10, 20 or 30 ms are valid
VAD_AGGRESSIVENESS = 2        # 0–3; higher = stricter noise rejection

# Segmenting: how much audio to collect before sending to Whisper.
MIN_SEGMENT_SECONDS = 0.5
MAX_SEGMENT_SECONDS = 8.0
MAX_SILENCE_BETWEEN_SPEECH_SECONDS = 0.8

# Set True to run noisereduce on each segment before Whisper (adds ~100 ms).
USE_DENOISER = False

# Whisper model size: "tiny" is fastest; "small" is more accurate for Finnish.
WHISPER_MODEL = "small"

# Initial prompt biases Whisper toward Finnish and English vocabulary.
WHISPER_PROMPT = (
    "Founderbot, hei botti, hello, moi, terve, kiitos, ole hyvä, "
    "what, how, why, yes, no, kyllä, ei."
)

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    audio_queue.put(indata.copy())


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Linear resample mono float32 array from from_sr to to_sr."""
    if from_sr == to_sr:
        return audio
    target_len = int(len(audio) * to_sr / from_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, target_len),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def float_to_int16_bytes(chunk: np.ndarray) -> bytes:
    """Convert float32 mono chunk to 16-bit PCM bytes for VAD."""
    return np.clip(chunk * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def transcribe(
    model: WhisperModel,
    frames: list[np.ndarray],
    native_sr: int,
) -> str:
    """Concatenate frames, resample to 16 kHz, optionally denoise, then Whisper."""
    audio = np.concatenate(frames, axis=0)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = resample(audio, native_sr, VAD_SAMPLE_RATE)
    if USE_DENOISER and nr is not None:
        try:
            audio = nr.reduce_noise(y=audio, sr=VAD_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            print(f"[Denoiser error] {exc}", file=sys.stderr)
    segments, info = model.transcribe(audio, task="transcribe", initial_prompt=WHISPER_PROMPT)
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

    native_sr = int(device_info["default_samplerate"])
    print(f"[Mic] {device_info['name']} — native {native_sr} Hz, VAD/Whisper at {VAD_SAMPLE_RATE} Hz")

    print(f"Loading Whisper model '{WHISPER_MODEL}'... (first run may take a moment)")
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # Capture one VAD frame worth of audio per callback (30 ms at native rate).
    capture_block = int(native_sr * VAD_FRAME_DURATION_MS / 1000)
    frame_seconds = VAD_FRAME_DURATION_MS / 1000.0
    # Pre-compute the expected 16 kHz frame length for VAD.
    vad_frame_len = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

    engine = ConversationEngine()

    speech_frames: list[np.ndarray] = []
    segment_duration = 0.0
    silence_duration = 0.0

    with sd.InputStream(
        samplerate=native_sr,
        blocksize=capture_block,
        device=None,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        print(f"[Engine] OFFLINE. Say '{engine.wake_word}' to wake me up.")
        try:
            while True:
                chunk = audio_queue.get()
                if chunk.size == 0:
                    continue

                # Mono
                mono = chunk[:, 0] if chunk.ndim > 1 else chunk

                # Resample chunk to 16 kHz for VAD
                mono_16k = resample(mono, native_sr, VAD_SAMPLE_RATE)
                # Trim/pad to exact VAD frame size
                if len(mono_16k) > vad_frame_len:
                    mono_16k = mono_16k[:vad_frame_len]
                elif len(mono_16k) < vad_frame_len:
                    mono_16k = np.pad(mono_16k, (0, vad_frame_len - len(mono_16k)))

                is_speech = vad.is_speech(float_to_int16_bytes(mono_16k), VAD_SAMPLE_RATE)

                if is_speech:
                    speech_frames.append(mono)  # store native-rate audio
                    segment_duration += frame_seconds
                    silence_duration = 0.0

                    if segment_duration >= MAX_SEGMENT_SECONDS:
                        text = transcribe(model, speech_frames, native_sr)
                        speech_frames.clear()
                        segment_duration = 0.0
                        if text:
                            engine.handle(text, now=time.monotonic())

                else:
                    if speech_frames:
                        silence_duration += frame_seconds
                        if silence_duration >= MAX_SILENCE_BETWEEN_SPEECH_SECONDS:
                            if segment_duration >= MIN_SEGMENT_SECONDS:
                                text = transcribe(model, speech_frames, native_sr)
                                if text:
                                    engine.handle(text, now=time.monotonic())
                            speech_frames.clear()
                            segment_duration = 0.0
                            silence_duration = 0.0

        except KeyboardInterrupt:
            print("\nStopping. Goodbye!")


if __name__ == "__main__":
    recognize_forever()
