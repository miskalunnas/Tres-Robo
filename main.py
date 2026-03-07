import json
import queue
import sys
from pathlib import Path

import sounddevice as sd
import vosk


MODEL_PATH = Path("models/vosk-model-small-en-us-0.15")

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


def audio_callback(indata, frames, time, status) -> None:
    """Callback from sounddevice whenever new audio data is available."""
    if status:
        print(f"[Audio status] {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))


def recognize_forever() -> None:
    """Continuously listen to the default microphone and print recognized text."""
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

    # Open an audio stream from the default input device
    with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        device=None,  # None = default input device
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        print("Listening... Speak into the microphone (Ctrl+C to stop).")
        try:
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result_json = recognizer.Result()
                    result = json.loads(result_json)
                    text = (result.get("text") or "").strip()
                    if text:
                        print(f"You said: {text}")
        except KeyboardInterrupt:
            print("\nStopping recognition. Goodbye!")


if __name__ == "__main__":
    recognize_forever()

