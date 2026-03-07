import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


# Wake word(s) that will bring the robot online.
WAKE_WORDS = ["hei botti"]

# How long (in seconds) of no recognized speech before going back to offline mode.
INACTIVITY_TIMEOUT_SECONDS = 9.0

# Minimum loudness in dB for audio to be treated as speech.
DB_THRESHOLD = 35.0

# How many seconds of loud audio we collect before sending it to Whisper.
MIN_CHUNK_SECONDS = 2.0

audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()


def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
    """Callback from sounddevice whenever new audio data is available."""
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    # Copy to avoid referencing the same buffer.
    audio_queue.put(indata.copy())


def compute_db(chunk: np.ndarray) -> float:
    """Return approximate loudness of the chunk in dB."""
    # Ensure mono.
    if chunk.ndim > 1:
        chunk = chunk[:, 0]
    rms = float(np.sqrt(np.mean(np.square(chunk), dtype=np.float64)))
    if rms <= 0:
        return -120.0
    # dB relative to full-scale (float32, range [-1, 1]).
    return 20.0 * np.log10(rms)


def transcribe_chunk(model: WhisperModel, audio: np.ndarray) -> str:
    """Run Whisper on a mono float32 audio buffer and return the combined text."""
    if audio.ndim > 1:
        audio = audio[:, 0]
    # WhisperModel will handle resampling internally.
    segments, _ = model.transcribe(audio, language="fi", task="transcribe")
    text_parts = [seg.text for seg in segments]
    return "".join(text_parts).strip()


def recognize_forever() -> None:
    """Continuously listen to the microphone with wake-word and dB threshold."""
    # Get default input device info (this should be your wired mic)
    try:
        device_info = sd.query_devices(kind="input")
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Could not query audio devices: {exc}", file=sys.stderr)
        sys.exit(1)

    samplerate = int(device_info["default_samplerate"])
    print(
        f"Using input device: {device_info['name']} "
        f"({samplerate} Hz sample rate)"
    )

    print("Loading Whisper model 'tiny' (Finnish)... This may take some time the first run.")
    model = WhisperModel("tiny", device="cpu", compute_type="int8")

    is_online = False
    last_activity_time = time.monotonic()

    print(
        f"Robot is OFFLINE. Say 'Hei botti' to wake it up. "
        f"(Ctrl+C to stop, ignoring sounds quieter than {DB_THRESHOLD:.0f} dB)"
    )

    # Buffer for loud audio that will be sent to Whisper.
    loud_buffer: list[np.ndarray] = []
    loud_buffer_duration = 0.0

    # Open an audio stream from the default input device
    with sd.InputStream(
        samplerate=samplerate,
        blocksize=1024,
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
                        "Going OFFLINE. Say 'Hei botti' to wake me up again."
                    )

                chunk = audio_queue.get()
                db = compute_db(chunk)
                if db < DB_THRESHOLD:
                    # Too quiet -> treat as noise, do not accumulate.
                    continue

                # Loud enough: accumulate into buffer.
                loud_buffer.append(chunk)
                loud_buffer_duration += len(chunk) / float(samplerate)

                if loud_buffer_duration < MIN_CHUNK_SECONDS:
                    # Not enough audio yet for a full recognition.
                    continue

                # We have collected enough loud audio -> send to Whisper.
                audio = np.concatenate(loud_buffer, axis=0)
                loud_buffer.clear()
                loud_buffer_duration = 0.0

                text = transcribe_chunk(model, audio)
                if not text:
                    continue

                normalized = text.lower()

                if not is_online:
                    # OFFLINE mode: print everything that is heard,
                    # but only switch to ONLINE when wake word is present.
                    print(f"[Offline heard] {text}")
                    if any(w in normalized for w in WAKE_WORDS):
                        is_online = True
                        last_activity_time = now
                        print("[Robot] Wake word detected: going ONLINE and listening.")
                    continue

                # ONLINE mode: react to all recognized speech and print it.
                last_activity_time = now
                print(f"You said: {text}")
                # Here is where you would later hook in:
                # - command parsing
                # - text-to-speech replies
                # - motor / servo control

        except KeyboardInterrupt:
            print("\nStopping recognition. Goodbye!")


if __name__ == "__main__":
    recognize_forever()

