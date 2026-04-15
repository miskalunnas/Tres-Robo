# Tres-Robo

A Raspberry Pi-based humanoid robot assistant for the TRES startup community at Robolabs, Hervanta.

## Architecture

```
OFFLINE  webrtcvad + Whisper (local, cheap)
         → wake word detected ("hei bot", "founderbot", ...)
ONLINE   Raw PCM audio streamed over WebSocket to Gemini Live
         Gemini handles speech-to-text, reasoning, and text-to-speech in one model
         Tool calls executed locally (music, vision, events, memory, Telegram...)
         → inactivity timeout OR end_conversation tool → back OFFLINE
```

The entry point is `main_gemini.py`.

## Hardware

| Component | Details |
|---|---|
| Board | Raspberry Pi 5 |
| Camera | Raspberry Pi AI Camera (IMX500, CSI) — falls back to Camera Module 2 or USB webcam |
| Microphone | Wired USB mic (`MIC_DEVICE=hw:2,0` or index) |
| Display | 800×480 HDMI screen (animated face) |
| Servo head | Pan: GPIO 12, Tilt: GPIO 13 (SG90 via lgpio) |
| Mute button | GPIO 17 (toggle mute) |
| PTT button | GPIO 27 (push-to-talk, expo mode) |

## Setup

### 1. System packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip portaudio19-dev ffmpeg
# For Pi Camera:
sudo apt install -y python3-picamera2 imx500-all
# raspi-config → Interface Options → Camera (enable)
```

### 2. Clone and create virtualenv

```bash
git clone <repo-url> Tres-Robo
cd Tres-Robo
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment variables

Copy `.env.example` to `.env` and fill in:

```bash
# Required
GOOGLE_API_KEY=...          # Gemini Live API key

# Optional overrides
GEMINI_LIVE_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
GEMINI_VOICE=Charon
GEMINI_TEMPERATURE=1.4

MIC_DEVICE=hw:2,0           # ALSA device string or PortAudio index
MIC_CHANNELS=1
MIC_SAMPLE_RATE=16000        # leave unset to auto-detect

# Music (yt-dlp)
YT_COOKIES_FILE=/path/to/cookies.txt   # YouTube Premium session (optional)

# Luma calendar (TRES events)
LUMA_API_KEY=...
LUMA_CALENDAR_ID=cal-...    # optional, auto-discovered from lu.ma/tres

# Telegram
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_MESSAGE_THREAD_ID=...   # optional, for forum topics

# Expo push-to-talk (temporary)
PTT_MODE=1                  # enable push-to-talk on GPIO 27

# GPIO pins (defaults shown)
MUTE_GPIO_PIN=17
PTT_GPIO_PIN=27
```

### 4. Run

```bash
DISPLAY=:0 python main_gemini.py
```

Say **"hei bot"**, **"founderbot"**, or any wake word to start a conversation.

---

## Tools

| Tool | What it does |
|---|---|
| `play_music` | Search and play music from YouTube by genre, artist, or song |
| `music_skip` | Skip to the next track |
| `music_pause` | Pause playback |
| `music_resume` | Resume paused playback |
| `music_stop` | Stop music and clear the queue |
| `music_add_to_queue` | Add a song/artist to the playback queue |
| `music_volume_up` | Increase volume |
| `music_volume_down` | Decrease volume |
| `get_events` | Fetch upcoming TRES events from the Luma calendar |
| `get_event_details` | Fetch full details for a specific event (description, location, link) |
| `get_menu` | Fetch today's lunch menus from Hervanta campus restaurants |
| `see` | Take a photo and answer a visual question |
| `lookup_knowledge` | Search local knowledge base for facts about TRES, people, slang |
| `save_knowledge` | Silently save a new fact to long-term memory |
| `telegram_send_message` | Stage a message to the TRES Telegram group (requires confirmation) |
| `confirm_action` | Confirm or cancel a staged Telegram message |
| `end_conversation` | End the session and go offline with a farewell |

---

## Memory

Long-term memory is stored in a local SQLite vector database (`memory/`).

- **Knowledge files** — static facts in `data/knowledge/` (TRES info, personas, house people). Loaded and embedded on startup.
- **Conversation facts** — the bot saves notable facts it hears via `save_knowledge`. A background curator (`memory/curator.py`) deduplicates and cleans these using GPT-4o-mini.
- **Retrieval** — `lookup_knowledge` uses semantic search (embeddings) so meaning-based queries find relevant facts even without exact keyword matches.

---

## Music playback

Uses `yt-dlp` to resolve search queries to audio URLs, then plays via `mpv` (preferred, supports volume ducking during speech) or `ffplay`.

```bash
sudo apt install mpv       # recommended
# or
sudo apt install ffmpeg    # provides ffplay
```

---

## Vision

The camera priority order:
1. **IMX500 AI Camera** — on-device EfficientDet person detection, used for face tracking
2. **Standard picamera2** — Camera Module 2 or other CSI camera
3. **OpenCV USB webcam** — dev fallback (`CV2_CAMERA_INDEX=0`)

The face tracker runs as a background thread and moves the pan/tilt servo head to follow the closest detected person. It automatically yields the camera to the `see` tool when needed.

---

## Telegram

The bot can send messages to a Telegram group but **always reads the message aloud and asks for voice confirmation first**.

1. Create a bot via [@BotFather](https://t.me/BotFather) and add it to your group.
2. Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`.
3. To find the chat ID: send a message in the group, then call `https://api.telegram.org/bot<TOKEN>/getUpdates` and look for `"chat":{"id": ...}`.

Group chat IDs are typically negative (e.g. `-1001234567890`).
