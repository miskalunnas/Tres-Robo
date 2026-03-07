import json
import queue
import sys
import time
from pathlib import Path

import sounddevice as sd
import vosk


MODEL_PATH = Path("models/vosk-model-small-en-us-0.15")

# Wake word(s) that will bring the robot online.
WAKE_WORDS = ["hei botti"]
# How long (in seconds) of no recognized speech before going back to offline mode.
INACTIVITY_TIMEOUT_SECONDS = 7.0

audio_queue: "queue.Queue[bytes]" = queue.Queue()


def ensure_model_exists() -> None:
    """Check that the Vosk model is available locally."""
    if not MODEL_PATH.exists():
        print(
            f"[ERROR] Speech model not found at '{MODEL_PATH}'.\n"
            "Download an English model from https://alphacephei.com/vosk/models\n"
            "and extract it so that this folder exists.\n"
            "Recommended: 'vosk-model-small-en-us-0.15'.",
            file=sys.stderr,
        )
        sys.exit(1)


def audio_callback(indata, frames, time_info, status) -> None:  # noqa: ARG001
    """Callback from sounddevice whenever new audio data is available."""
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))


def recognize_forever() -> None:
    """Continuously listen to the microphone with wake-word controlled online/offline modes."""
    ensure_model_exists()

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

    print(f"Loading speech model from '{MODEL_PATH}' (this may take a few seconds)...")
    model = vosk.Model(str(MODEL_PATH))
    recognizer = vosk.KaldiRecognizer(model, samplerate)

    is_online = False
    last_activity_time = time.monotonic()

    print("Robot is OFFLINE. Say 'Hei botti' to wake it up. (Ctrl+C to stop)")

    # Open an audio stream from the default input device
    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        device=None,  # None = default input device
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        try:
            while True:
                # Check for inactivity timeout when online
                now = time.monotonic()
                if is_online and (now - last_activity_time) >= INACTIVITY_TIMEOUT_SECONDS:
                    is_online = False
                    print(
                        f"[Robot] No speech for {INACTIVITY_TIMEOUT_SECONDS:.0f} seconds. "
                        "Going OFFLINE. Say 'Hei botti' to wake me up again."
                    )

                data = audio_queue.get()
                if not recognizer.AcceptWaveform(data):
                    # Not a complete phrase yet; keep listening.
                    continue

                result_json = recognizer.Result()
                result = json.loads(result_json)
                text = (result.get("text") or "").strip()
                if not text:
                    continue

                normalized = text.lower()

                if not is_online:
                    # OFFLINE mode: only look for the wake word.
                    if any(w in normalized for w in WAKE_WORDS):
                        is_online = True
                        last_activity_time = now
                        print("[Robot] Wake word detected: going ONLINE and listening.")
                    # Ignore everything else while offline.
                    continue

                # ONLINE mode: react to all recognized speech.
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

