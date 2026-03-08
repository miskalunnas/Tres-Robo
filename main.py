import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd
import webrtcvad

try:
    import noisereduce as nr
except Exception:  # noqa: BLE001
    nr = None

from conversation import ConversationEngine
from voice.tts import is_busy

# Puheen tunnistus: True = OpenAI Whisper API (pilvi), False = paikallinen faster-whisper.
USE_CLOUD_STT = True

if USE_CLOUD_STT:
    from voice import stt_openai
else:
    from faster_whisper import WhisperModel

# Rate expected by webrtcvad and Whisper. Capture may differ — we resample.
VAD_SAMPLE_RATE = 16_000

# VAD: 0 = herkimmin, 1 = tasapaino, 2 = vähemmän taustamelua puheena, 3 = vahvin.
VAD_AGGRESSIVENESS = 2

# VAD frame length (10, 20 or 30 ms).
VAD_FRAME_DURATION_MS = 30

# Segmenting: how much audio to collect before sending to Whisper.
MIN_SEGMENT_SECONDS = 0.5
MAX_SEGMENT_SECONDS = 8.0
# OFFLINE: herätys sujuvampi — lyhyet "hei bot" pääsevät läpi.
MIN_SEGMENT_WHEN_OFFLINE = 0.4
# ONLINE + musiikki: korkeampi kynnys = botti ei keskeytä musiikkia lyhyellä puheella.
MIN_SEGMENT_WHEN_MUSIC_PLAYING = 1.2
# Hiljaisuus ennen kuin lähetetään Whisperille: isompi = botti ei puhu päälle, pienempi = nopeampi vastaus.
MAX_SILENCE_BETWEEN_SPEECH_SECONDS = 0.9

# Interruption capture: botti kuuntelee puhuessaan, päättää reagoidaanko.
# MIN = vähimmäisaika ennen lähetystä; MAX = puskurin koko — korkea MAX = koko puhe bottille.
INTERRUPT_MIN_SEGMENT_SECONDS = 1.8
INTERRUPT_MAX_SEGMENT_SECONDS = 8.0  # Sama kuin normaali — koko alusta lähtien kuunneltu puhe saatavilla
INTERRUPT_MAX_SILENCE_BETWEEN_SPEECH_SECONDS = 0.5

# Musiikin ducking: hiljaisuuden kesto (s) puheen jälkeen ennen volyymin palautusta.
MUSIC_UNDUCK_SILENCE_SECONDS = 2.5

# Set True to run noisereduce on each segment before Whisper (adds ~100 ms).
USE_DENOISER = False

# Whisper model: "tiny" = nopein, "base" = nopea kompromissi, "small"/"medium" = tarkempi suomeen.
WHISPER_MODEL = "small"
# Whisper: herätyssanat ensin (tärkein kauempaa puhuttaessa). Fraasit auttavat tunnistusta.
WHISPER_PROMPT = (
    "Founderbot, founderbott, founder bot, found a bot, founder bott, founderbotti. "
    "Hei botti, hei bot, hei robot, hei robotti. Kuule botti, kuule bot, bot kuule, botti kuule. "
    "Terve botti, terve bot, moro botti, moro bot, moi botti, moi bot. Ok bot, okay bot, yo bot. "
    "Soita, soita musiikki, play, skip, seuraava, tauko, pause, jatka, resume, lopeta, stop, queue, poista kaikki, clear. "
    "Volume, volume up, volume down, louder, quieter, kovempaa, hiljempaa, ääni ylös, ääni alas. "
    "Kello, mitä kello on, paljonko kello, aika, time. Vitsi, kerro vitsi, joke. "
    "Ruokalista, lounaslista, mitä ruokana, reaktori, newton, menu, lunch. "
    "Moi, hei, moro, terve, moikka, päivää, kiitos, joo, ei, kyllä, selvä. "
    "Miten voit, pöhinä, pöhinää, voisitko, kerro, mitä, miksi, milloin, missä, mikä, kuka, apua, mitä osaat. "
    "Näkemiin, hei hei, moi moi, goodbye, bye, siinä kaikki, ei muuta, palataan myöhemmin, mene lepotilaan."
)


@dataclass
class SegmentTask:
    frames: list[np.ndarray]
    heard_at: float
    interruption: bool = False


@dataclass
class UtteranceTask:
    text: str
    heard_at: float
    interruption: bool = False
    language: str = ""


audio_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
segment_queue: "queue.Queue[SegmentTask]" = queue.Queue(maxsize=8)
utterance_queue: "queue.Queue[UtteranceTask]" = queue.Queue(maxsize=8)


def _put_latest(q: queue.Queue, item) -> None:
    """Keep queues bounded by dropping the oldest pending item when full."""
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
    _put_latest(audio_queue, indata.copy())


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


def _prepare_audio(frames: list[np.ndarray], native_sr: int) -> np.ndarray:
    """Concatenate frames, resample to 16 kHz, optionally denoise, normalize. Returns mono float32."""
    audio = np.concatenate(frames, axis=0)
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = resample(audio, native_sr, VAD_SAMPLE_RATE)
    if USE_DENOISER and nr is not None:
        try:
            audio = nr.reduce_noise(y=audio, sr=VAD_SAMPLE_RATE)
        except Exception as exc:  # noqa: BLE001
            print(f"[Denoiser error] {exc}", file=sys.stderr)
    peak = np.abs(audio).max()
    if peak > 1e-4:
        # Kauempaa puhuttaessa ääni on hiljaisempi — vahvistus jopa 5x
        audio = audio * min(0.95 / peak, 5.0)
    return audio


def _transcribe_cloud(audio: np.ndarray) -> tuple[str, str]:
    """OpenAI Whisper API (pilvi). Palauttaa (teksti, kielikoodi)."""
    t0 = time.monotonic()
    text, lang = stt_openai.transcribe(
        audio,
        VAD_SAMPLE_RATE,
        language=None,  # auto-detect
        prompt=WHISPER_PROMPT,
        return_language=True,
    )
    print(f"[Timing] Whisper cloud ({len(audio)/VAD_SAMPLE_RATE:.1f}s): {time.monotonic()-t0:.2f}s")
    if text:
        print(f"[Whisper/{lang or '?'}] {text}")
    return text, lang


def _transcribe_local_with_lang(model, audio: np.ndarray) -> tuple[str, str]:
    """Paikallinen faster-whisper, palauttaa (teksti, kielikoodi)."""
    t0 = time.monotonic()
    segments, info = model.transcribe(
        audio,
        task="transcribe",
        language=None,  # auto-detect
        initial_prompt=WHISPER_PROMPT,
        condition_on_previous_text=True,
        no_speech_threshold=0.35,
        temperature=(0.0, 0.2, 0.4),
    )
    text = "".join(seg.text for seg in segments).strip()
    lang = getattr(info, "language", "") or ""
    print(f"[Timing] Whisper local ({len(audio)/VAD_SAMPLE_RATE:.1f}s): {time.monotonic()-t0:.2f}s")
    if text:
        print(f"[Whisper/{lang or '?'}] {text}")
    return text, lang


def transcribe(
    model,
    frames: list[np.ndarray],
    native_sr: int,
) -> tuple[str, str]:
    """Prepare audio and run STT. Returns (text, language_code)."""
    audio = _prepare_audio(frames, native_sr)
    if np.abs(audio).max() < 0.05:
        return "", ""
    if USE_CLOUD_STT:
        return _transcribe_cloud(audio)
    return _transcribe_local_with_lang(model, audio)


def _transcription_worker(model, native_sr: int) -> None:
    while True:
        task = segment_queue.get()
        try:
            text, lang = transcribe(model, task.frames, native_sr)
            if text:
                _put_latest(
                    utterance_queue,
                    UtteranceTask(
                        text=text,
                        heard_at=task.heard_at,
                        interruption=task.interruption,
                        language=lang,
                    ),
                )
        finally:
            segment_queue.task_done()


def _dialogue_worker(engine: ConversationEngine) -> None:
    while True:
        utterance = utterance_queue.get()
        try:
            if utterance.interruption:
                engine.handle_interruption(
                    utterance.text,
                    now=utterance.heard_at,
                    language=utterance.language,
                )
            else:
                engine.handle(
                    utterance.text,
                    now=utterance.heard_at,
                    language=utterance.language,
                )
        finally:
            utterance_queue.task_done()


def listen_forever() -> None:
    # MIC_DEVICE: ALSA device string or sounddevice index for the USB mic.
    # PortAudio sometimes misreports USB capture devices as 0 input channels
    # even when arecord confirms they work. Using the ALSA hw string bypasses this.
    MIC_DEVICE = "hw:2,0"
    native_sr = 48_000  # confirmed via sd.query_devices()
    print(f"[Mic] {MIC_DEVICE} — native {native_sr} Hz, VAD/STT at {VAD_SAMPLE_RATE} Hz")

    if USE_CLOUD_STT:
        print("[STT] Using OpenAI Whisper (cloud). Set OPENAI_API_KEY in .env")
        model = None
    else:
        print(f"Loading Whisper model '{WHISPER_MODEL}'... (first run may take a moment)")
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    # Capture one VAD frame worth of audio per callback (30 ms at native rate).
    capture_block = int(native_sr * VAD_FRAME_DURATION_MS / 1000)
    frame_seconds = VAD_FRAME_DURATION_MS / 1000.0
    # Pre-compute the expected 16 kHz frame length for VAD.
    vad_frame_len = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000)

    engine = ConversationEngine()
    try:
        from Tools.music import check_music_ready

        check_music_ready()
    except Exception:
        pass

    threading.Thread(
        target=_transcription_worker,
        args=(model, native_sr),
        daemon=True,
    ).start()
    threading.Thread(
        target=_dialogue_worker,
        args=(engine,),
        daemon=True,
    ).start()

    identity_watcher = None
    # Vision (kasvontunnistus) pois päältä — säästää CPU:ta.

    speech_frames: list[np.ndarray] = []
    segment_duration = 0.0
    silence_duration = 0.0
    last_speech_time = time.monotonic()
    last_speech_or_send = 0.0  # ducking: unduck after this long silence
    music_ducked = False
    capture_mode = "normal"

    with sd.InputStream(
        samplerate=native_sr,
        blocksize=capture_block,
        device=MIC_DEVICE,
        dtype="float32",
        channels=1,
        callback=audio_callback,
    ):
        print(f"[Engine] OFFLINE. Say 'founderbot', 'hei bot' or 'kuule bot' to wake me up.")
        print("[Engine] When ONLINE: music (play, skip, pause, resume, stop), menu, time, joke.")
        try:
            while True:
                now = time.monotonic()
                chunk = audio_queue.get()
                if chunk.size == 0:
                    continue

                current_mode = "interrupt" if is_busy() else "normal"
                if current_mode != capture_mode:
                    speech_frames.clear()
                    segment_duration = 0.0
                    silence_duration = 0.0
                    capture_mode = current_mode

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
                    last_speech_time = time.monotonic()
                    last_speech_or_send = last_speech_time
                    # Kun soittaa musiikkia ja puhutaan, pienennä volyymiä (ducking).
                    try:
                        from Tools.music import is_playing, duck
                        if is_playing() and not music_ducked:
                            duck()
                            music_ducked = True
                    except Exception:
                        pass

                    max_segment = (
                        INTERRUPT_MAX_SEGMENT_SECONDS
                        if capture_mode == "interrupt"
                        else MAX_SEGMENT_SECONDS
                    )
                    if segment_duration >= max_segment:
                        _put_latest(
                            segment_queue,
                            SegmentTask(
                                frames=list(speech_frames),
                                heard_at=last_speech_time,
                                interruption=(capture_mode == "interrupt"),
                            ),
                        )
                        last_speech_or_send = time.monotonic()
                        speech_frames.clear()
                        segment_duration = 0.0

                else:
                    if speech_frames:
                        silence_duration += frame_seconds
                        max_silence = (
                            INTERRUPT_MAX_SILENCE_BETWEEN_SPEECH_SECONDS
                            if capture_mode == "interrupt"
                            else MAX_SILENCE_BETWEEN_SPEECH_SECONDS
                        )
                        if capture_mode == "interrupt":
                            min_segment = INTERRUPT_MIN_SEGMENT_SECONDS
                        else:
                            try:
                                from Tools.music import is_playing
                                if is_playing() and engine.is_online():
                                    min_segment = MIN_SEGMENT_WHEN_MUSIC_PLAYING
                                elif not engine.is_online():
                                    min_segment = MIN_SEGMENT_WHEN_OFFLINE
                                else:
                                    min_segment = MIN_SEGMENT_SECONDS
                            except Exception:
                                min_segment = MIN_SEGMENT_SECONDS
                        if silence_duration >= max_silence:
                            if segment_duration >= min_segment:
                                _put_latest(
                                    segment_queue,
                                    SegmentTask(
                                        frames=list(speech_frames),
                                        heard_at=last_speech_time,
                                        interruption=(capture_mode == "interrupt"),
                                    ),
                                )
                                last_speech_or_send = time.monotonic()
                            speech_frames.clear()
                            segment_duration = 0.0
                            silence_duration = 0.0
                    # Unduck musiikki kun hiljaisuus on tarpeeksi pitkä puheen jälkeen.
                    if music_ducked and (now - last_speech_or_send) >= MUSIC_UNDUCK_SILENCE_SECONDS:
                        try:
                            from Tools.music import unduck
                            unduck()
                        except Exception:
                            pass
                        music_ducked = False

        except KeyboardInterrupt:
            if identity_watcher is not None:
                identity_watcher.stop()
            print("\nStopping. Goodbye!")


if __name__ == "__main__":
    listen_forever()
