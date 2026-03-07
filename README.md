# Tres-Robo

Robot head project using a Raspberry Pi 5 with a wired microphone and local voice recognition.

## What this repo contains

- `main.py`: Python script that listens to the default microphone and prints recognized speech.
- `requirements.txt`: Python dependencies for audio input and offline speech recognition.

The code is written in **Python** and is intended to run on your **Raspberry Pi 5** (it will also run on your PC for testing, as long as you have a microphone).

## Setup on Raspberry Pi 5

1. **Update system packages**

   ```bash
   sudo apt update
   sudo apt upgrade -y
   ```

2. **Install Python and audio tools**

   (Most Raspberry Pi OS images already have Python 3 installed.)

   ```bash
   sudo apt install -y python3 python3-venv python3-pip portaudio19-dev
   ```

3. **Clone this repository onto your Pi**

   ```bash
   git clone <your-repo-url> tres-robo
   cd tres-robo
   ```

4. **Create and activate a virtual environment (recommended)**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. **Install Python dependencies (Whisper, VAD, etc.)**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   The first time you run the script, the **Whisper "tiny" model** will be downloaded automatically. It is multilingual, but the code forces the language to **Finnish (`fi`)**.

## Running the voice recognition script

1. Plug your **wired microphone** into the Raspberry Pi.
2. Make sure you are in the project folder (and that your virtual environment is active if you created one).
3. Run:

   ```bash
   python3 main.py
   ```

You should see something like:

- Information about the **input device** and sample rate.
- A message that the **Whisper model** is loading.
- A message that the robot is **OFFLINE** and waiting for the wake word "Hei botti".
- As you speak, the audio is first filtered by **WebRTC VAD** (voice activity detection) to ignore pure background noise, then sent in small segments to Whisper.
  - In OFFLINE mode, recognized speech segments are printed as: `[Offline heard] ...` and are only used to detect the wake word.
  - When you say "Hei botti", the robot goes ONLINE.
  - In ONLINE mode, segments are printed as: `You said: ...`

### Noise handling

- The microphone stream is filtered by **WebRTC VAD** (`webrtcvad`), which tries to keep only speech and drop pure noise.
- There is also an **optional denoiser** (`noisereduce`) wired into the code. It is disabled by default for performance reasons; you can enable it by setting `USE_DENOISER = True` in `main.py` if you want more aggressive noise reduction.\
  
## Next steps

- Connect the recognized text (e.g. "turn left", "look up", "blink") to **motor control** or other actions for your robot head.
- Add a simple command parser in `main.py` that looks for certain keywords and triggers GPIO or serial commands to your robot hardware.

This repository is a starting point: it gives you working microphone input and speech-to-text so you can focus on the robot behavior.