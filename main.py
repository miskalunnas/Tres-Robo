import queue
import sys
import time
from typing import List, Optional

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

try:
    import noisereduce as nr
except Exception:  # noqa: BLE001
    nr = None


# Wake word(s) that will bring the robot online.
WAKE_WORDS = ["founderbot"]

# Target sample rate for both VAD and Whisper input.
TARGET_SAMPLE_RATE = 48_000

# How long (in seconds) of no recognized speech before going back to offline mode.
INACTIVITY_TIMEOUT_SECONDS = 9.0

# Voice activity detection settings.
VAD_FRAME_DURATION_MS = 30  # 10, 20 or 30 ms are supported by WebRTC VAD
VAD_AGGRESSIVENESS = 2  # 0–3, higher = more noise rejection

# Segmenting settings.
MIN_SEGMENT_SECONDS = 1.0
MAX_SEGMENT_SECONDS = 5.0
MAX_SILENCE_BETWEEN_SPEECH_SECONDS = 0.5

# Optional denoiser toggle. When True and noisereduce is installed,
# each speech segment will be run through noise reduction before Whisper.
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
    """Callback from sounddevice whenever new audio data is available."""
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    # Copy to avoid referencing the same buffer.
    audio_queue.put(indata.copy())


def transcribe_chunk(model: WhisperModel, audio: np.ndarray) -> str:
    """Run Whisper on a mono float32 audio buffer and return the combined text."""
    if audio.ndim > 1:
        audio = audio[:, 0]
    # Optionally apply denoising.
    if USE_DENOISER and nr is not None and audio.size > 0:
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

    default_samplerate = int(device_info["default_samplerate"])
    print(
        f"Using input device: {device_info['name']} "
        f"(default {default_samplerate} Hz, capturing at {TARGET_SAMPLE_RATE} Hz)"
    )

    print("Loading Whisper model 'tiny'... (first run may take a moment)")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    # Set up WebRTC VAD.
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_length = int(TARGET_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

    is_online = False
    last_activity_time = time.monotonic()

    print("Robot is OFFLINE. Say 'founderbot' to wake it up. (Ctrl+C to stop)")

    # Buffers for current speech segment.
    speech_frames: list[np.ndarray] = []
    segment_duration = 0.0
    silence_during_segment = 0.0

    # Open an audio stream from the default input device
    with sd.InputStream(
        samplerate=TARGET_SAMPLE_RATE,
        blocksize=frame_length,
        device=None,  # None = default input device
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        try:
            while True:
                now = time.monotonic()

                # Check for inactivity timeout when online
                if is_online and (now - last_activity_time) >= INACTIVITY_TIMEOUT_SECONDS:
                    is_online = False
                    print(
                        f"[Robot] No speech for {INACTIVITY_TIMEOUT_SECONDS:.0f} seconds. "
                        "Going OFFLINE. Say 'founderbot' to wake me up again."
                    )

                chunk = audio_queue.get()
                if chunk.size == 0:
                    continue

                pcm_bytes = float_chunk_to_int16_bytes(chunk)
                is_speech = vad.is_speech(pcm_bytes, TARGET_SAMPLE_RATE)
                frame_seconds = len(chunk) / float(TARGET_SAMPLE_RATE)

                if is_speech:
                    speech_frames.append(chunk)
                    segment_duration += frame_seconds
                    silence_during_segment = 0.0

                    if segment_duration >= MAX_SEGMENT_SECONDS:
                        _, is_online, last_activity_time = process_segment(
                            model,
                            speech_frames,
                            now,
                            is_online,
                            last_activity_time,
                        )
                        speech_frames.clear()
                        segment_duration = 0.0
                        silence_during_segment = 0.0
                else:
                    if speech_frames:
                        silence_during_segment += frame_seconds
                        if (
                            silence_during_segment >= MAX_SILENCE_BETWEEN_SPEECH_SECONDS
                            or segment_duration >= MAX_SEGMENT_SECONDS
                        ):
                            if segment_duration >= MIN_SEGMENT_SECONDS:
                                _, is_online, last_activity_time = process_segment(
                                    model,
                                    speech_frames,
                                    now,
                                    is_online,
                                    last_activity_time,
                                )
                            speech_frames.clear()
                            segment_duration = 0.0
                            silence_during_segment = 0.0

        except KeyboardInterrupt:
            print("\nStopping recognition. Goodbye!")


if __name__ == "__main__":
    listen_forever()

