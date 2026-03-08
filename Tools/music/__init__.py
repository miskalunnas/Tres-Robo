"""Music playback via yt-dlp and external player (ffplay / mpv).

The bot is intended to use a YouTube Premium account for playback (better quality, no ads).
Ducking: when someone talks while music plays, volume is lowered (mpv only; ffplay has no runtime volume).

Requirements:
  - Python: yt-dlp (pip install yt-dlp; see requirements.txt).
  - System: ffplay (from ffmpeg) or mpv. On Raspberry Pi: sudo apt install ffmpeg.
  - Ducking: mpv recommended (ffplay cannot change volume during playback).
  - .env: YT_COOKIES_FILE=/path/to/cookies.txt (YouTube Premium session).
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Path to cookies.txt for YouTube Premium (default: bot uses this account for playback)
_YT_COOKIES_FILE = os.getenv("YT_COOKIES_FILE", "").strip()


def check_music_ready() -> bool:
    """Check if yt-dlp and a player (ffplay/mpv) are available. Print a one-line status. Returns True if ready."""
    yt_dlp_ok = False
    try:
        result = subprocess.run(
            [sys.executable, "-m", "yt_dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        yt_dlp_ok = result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    player = None
    for cmd in ("mpv", "ffplay"):
        if shutil.which(cmd):
            player = cmd
            break

    if yt_dlp_ok and player:
        msg = f"[Music] Ready (yt-dlp + {player})."
        if _YT_COOKIES_FILE and os.path.isfile(_YT_COOKIES_FILE):
            msg += " YouTube Premium (YT_COOKIES_FILE)."
        else:
            msg += " Set YT_COOKIES_FILE in .env for YouTube Premium."
        print(msg)
        return True

    parts = []
    if not yt_dlp_ok:
        parts.append("pip install yt-dlp")
    if not player:
        parts.append("sudo apt install ffmpeg  OR  sudo apt install mpv")
    print(f"[Music] Not ready for playback: {'; '.join(parts)}.", file=sys.stderr)
    return False


class MusicPlayer:
    """Manages a playback queue and the current player subprocess."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._process: subprocess.Popen | None = None
        self._paused: bool = False
        self._stopped: bool = False
        self._volume: int = 80  # 0–100, käytetään seuraavassa soittokerrassa
        self._current_query: str | None = None
        self._lock = threading.Lock()
        self._watcher: threading.Thread | None = None
        self._player_cmd: str | None = None
        self._ipc_socket_path: str | None = None  # mpv IPC (ducking)
        self._ducked: bool = False
        self._saved_volume_before_duck: int = 80
        self._detect_player()

    # ------------------------------------------------------------------
    # Player detection
    # ------------------------------------------------------------------

    def _detect_player(self) -> None:
        # Prefer mpv (ducking via IPC); fallback to ffplay
        for cmd in ("mpv", "ffplay"):
            if shutil.which(cmd):
                self._player_cmd = cmd
                return
        self._player_cmd = None

    @staticmethod
    def _mpv_ipc_path() -> str:
        """Path/name for mpv IPC. Unix: socket path; Windows: pipe name (mpv creates \\\\.\\pipe\\<name>)."""
        pid = os.getpid()
        if os.name == "nt":
            return "mpv-bot-" + str(pid)
        return "/tmp/mpv-bot-" + str(pid)

    # ------------------------------------------------------------------
    # URL resolution (yt-dlp)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_url(query: str) -> str | None:
        """Use yt-dlp to turn a search query into a direct audio URL."""
        cmd = [
            sys.executable, "-m", "yt_dlp",
            "-g", "-f", "bestaudio/best",
            "--no-playlist",
            "--default-search", "ytsearch",
            f"ytsearch1:{query}",
        ]
        if _YT_COOKIES_FILE and os.path.isfile(_YT_COOKIES_FILE):
            cmd.extend(["--cookies", _YT_COOKIES_FILE])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            print("[Music] Search timed out.", file=sys.stderr)
            return None
        except FileNotFoundError:
            print("[Music] yt-dlp not found. Install with: pip install yt-dlp", file=sys.stderr)
            return None

        if result.returncode != 0 or not (result.stdout or "").strip():
            print(f"[Music] No result for: {query!r}", file=sys.stderr)
            return None
        return result.stdout.strip()

    # ------------------------------------------------------------------
    # Subprocess launch
    # ------------------------------------------------------------------

    def _start_player(self, url: str, *, use_ipc: bool = True) -> subprocess.Popen | None:
        if not self._player_cmd:
            print(
                "[Music] No player found. Install ffmpeg or mpv: sudo apt install ffmpeg  OR  sudo apt install mpv",
                file=sys.stderr,
            )
            return None

        if self._player_cmd == "ffplay":
            self._ipc_socket_path = None
            # ffplay -volume: 0–256 (256 = 100%), not 0–1
            vol = max(0, min(256, int(self._volume / 100.0 * 256)))
            args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", str(vol), url]
        else:
            # mpv: IPC vain duckingia varten; jos käynnistys epäonnistuu, yritä ilman IPC
            if use_ipc:
                self._ipc_socket_path = self._mpv_ipc_path()
                args = [
                    "mpv", "--no-video", "--really-quiet",
                    "--volume=" + str(self._volume),
                    "--input-ipc-server=" + self._ipc_socket_path,
                    url,
                ]
            else:
                self._ipc_socket_path = None
                args = [
                    "mpv", "--no-video", "--really-quiet",
                    "--volume=" + str(self._volume),
                    url,
                ]

        kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if flags:
                kwargs["creationflags"] = flags
        else:
            kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(args, **kwargs)
            if self._player_cmd == "mpv" and use_ipc:
                time.sleep(0.3)  # IPC socket/pipe ready
                if proc.poll() is not None:
                    # mpv lähti pois heti (esim. IPC ei tuettu) → yritä ilman IPC
                    return self._start_player(url, use_ipc=False)
            return proc
        except Exception as e:
            print(f"[Music] Failed to start player: {e}", file=sys.stderr)
            return None

    # ------------------------------------------------------------------
    # Watcher thread: waits for the process to finish, then plays next
    # ------------------------------------------------------------------

    def _watch_process(self) -> None:
        """Block until the current process exits, then auto-play next."""
        proc = self._process
        if proc is None:
            return
        proc.wait()

        with self._lock:
            if self._stopped:
                return
            self._process = None
            self._paused = False

        self._play_next_from_queue()

    def _start_watcher(self) -> None:
        self._watcher = threading.Thread(target=self._watch_process, daemon=True)
        self._watcher.start()

    # ------------------------------------------------------------------
    # Internal: play next item from queue
    # ------------------------------------------------------------------

    def _play_next_from_queue(self) -> None:
        with self._lock:
            if not self._queue:
                print("[Music] Queue is empty.")
                return
            query = self._queue.popleft()
            self._current_query = query

        print(f"[Music] Resolving: {query!r} ...")
        url = self._resolve_url(query)
        if url is None:
            self._play_next_from_queue()
            return

        with self._lock:
            self._process = self._start_player(url)
            self._paused = False
            self._stopped = False

        if self._process:
            print(f"[Music] Playing: {query!r}")
            self._start_watcher()
        else:
            self._play_next_from_queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, query: str) -> bool:
        """Stop current playback, clear queue, and play *query* immediately."""
        query = (query or "music").strip() or "music"
        self._kill_current()

        with self._lock:
            self._queue.clear()
            self._stopped = False

        print(f"[Music] Resolving: {query!r} ...")
        url = self._resolve_url(query)
        if url is None:
            return False

        with self._lock:
            self._current_query = query
            self._process = self._start_player(url)
            self._paused = False

        if self._process is None:
            return False

        print(f"[Music] Playing: {query!r}")
        self._start_watcher()
        return True

    def play_async(self, query: str) -> None:
        """Start playback in a background thread so the caller can return immediately (e.g. for TTS)."""
        def _run() -> None:
            self.play(query)
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def add_to_queue(self, query: str) -> None:
        """Append a query to the playback queue."""
        query = (query or "").strip()
        if not query:
            print("[Music] No query to add.")
            return
        with self._lock:
            self._queue.append(query)
        print(f"[Music] Added to queue: {query!r}  (queue length: {len(self._queue)})")

    def skip(self) -> bool:
        """Skip the current track and play the next from the queue. Returns True if something was playing."""
        with self._lock:
            if self._process is None:
                print("[Music] Nothing playing to skip.")
                return False
            self._stopped = False
        print("[Music] Skipping...")
        self._kill_current()
        return True

    def pause(self) -> bool:
        """Pause the current player process. Returns True if paused."""
        with self._lock:
            if self._process is None or self._paused:
                print("[Music] Nothing to pause.")
                return False
            self._paused = True
            pid = self._process.pid

        if os.name == "nt":
            self._suspend_windows(pid)
        else:
            os.kill(pid, signal.SIGSTOP)
        print("[Music] Paused.")
        return True

    def resume(self) -> bool:
        """Resume the paused player process. Returns True if resumed."""
        with self._lock:
            if self._process is None or not self._paused:
                print("[Music] Nothing to resume.")
                return False
            self._paused = False
            pid = self._process.pid

        if os.name == "nt":
            self._resume_windows(pid)
        else:
            os.kill(pid, signal.SIGCONT)
        print("[Music] Resumed.")
        return True

    def stop(self) -> bool:
        """Stop playback and clear the queue. Returns True if something was stopped."""
        with self._lock:
            had_process = self._process is not None
            self._queue.clear()
            self._stopped = True
        self._kill_current()
        print("[Music] Stopped and queue cleared.")
        return had_process

    def volume_up(self) -> int:
        """Increase volume by 10 (0–100). Uses mpv IPC when available; otherwise restarts. Returns new volume."""
        with self._lock:
            self._volume = min(100, self._volume + 10)
            if self._ducked:
                self._saved_volume_before_duck = self._volume
            if self._process and self._player_cmd == "mpv" and self._ipc_socket_path:
                vol = self._volume
            else:
                vol = None
            if vol is None and self._process and self._current_query:
                self._queue.appendleft(self._current_query)
        if vol is not None:
            self._mpv_send_volume(vol)
            print(f"[Music] Volume up → {self._volume}%")
            return self._volume
        self._kill_current()
        print(f"[Music] Volume up → {self._volume}%")
        return self._volume

    def volume_down(self) -> int:
        """Decrease volume by 10 (0–100). Uses mpv IPC when available; otherwise restarts. Returns new volume."""
        with self._lock:
            self._volume = max(0, self._volume - 10)
            if self._ducked:
                self._saved_volume_before_duck = self._volume
            if self._process and self._player_cmd == "mpv" and self._ipc_socket_path:
                vol = self._volume
            else:
                vol = None
            if vol is None and self._process and self._current_query:
                self._queue.appendleft(self._current_query)
        if vol is not None:
            self._mpv_send_volume(vol)
            print(f"[Music] Volume down → {self._volume}%")
            return self._volume
        self._kill_current()
        print(f"[Music] Volume down → {self._volume}%")
        return self._volume

    def get_volume(self) -> int:
        """Return current volume 0–100."""
        return self._volume

    def is_playing(self) -> bool:
        """True if music is currently playing (process running)."""
        with self._lock:
            return self._process is not None and not self._paused

    DUCK_VOLUME = 12  # Volume % when ducked (puhe kuuluu paremmin)

    def _mpv_send_volume(self, volume: int) -> bool:
        """Send volume to mpv via IPC. Returns True on success."""
        path = self._ipc_socket_path
        if not path or self._player_cmd != "mpv":
            return False
        try:
            msg = json.dumps({"command": ["set_property", "volume", volume]}) + "\n"
            if os.name == "nt":
                # Windows: named pipe \\.\pipe\<name>
                pipe_path = r"\\.\pipe" + "\\" + path
                with open(pipe_path, "wb") as f:
                    f.write(msg.encode("utf-8"))
            else:
                import socket
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(path)
                sock.sendall(msg.encode("utf-8"))
                sock.close()
            return True
        except Exception as e:
            print(f"[Music] IPC volume failed: {e}", file=sys.stderr)
            return False

    def duck(self) -> None:
        """Lower volume a lot so speech is heard (mpv only). No-op if ffplay or not playing."""
        with self._lock:
            if self._process is None or self._player_cmd != "mpv" or self._ducked:
                return
            self._saved_volume_before_duck = self._volume
            self._ducked = True
        self._mpv_send_volume(self.DUCK_VOLUME)
        print(f"[Music] Ducking → {self.DUCK_VOLUME}%")

    def unduck(self) -> None:
        """Restore volume after ducking (mpv only)."""
        with self._lock:
            if not self._ducked:
                return
            self._ducked = False
            vol = self._saved_volume_before_duck
        self._mpv_send_volume(vol)
        print(f"[Music] Unduck → {vol}%")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kill_current(self) -> None:
        with self._lock:
            proc = self._process
            if proc is None:
                return
            self._paused = False
            self._ipc_socket_path = None
            self._ducked = False

        if os.name != "nt":
            try:
                os.kill(proc.pid, signal.SIGCONT)
            except OSError:
                pass

        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

        with self._lock:
            if self._process is proc:
                self._process = None

    # Windows suspend / resume via kernel32
    @staticmethod
    def _suspend_windows(pid: int) -> None:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            handle = kernel32.OpenProcess(0x1F0FFF, False, pid)
            if handle:
                kernel32.DebugActiveProcess(pid)
                kernel32.CloseHandle(handle)
        except Exception as e:
            print(f"[Music] Windows suspend failed: {e}", file=sys.stderr)

    @staticmethod
    def _resume_windows(pid: int) -> None:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.DebugActiveProcessStop(pid)
        except Exception as e:
            print(f"[Music] Windows resume failed: {e}", file=sys.stderr)


# ------------------------------------------------------------------
# Module-level singleton and public functions
# ------------------------------------------------------------------

_player = MusicPlayer()


def play(query: str) -> bool:
    return _player.play(query)


def play_async(query: str) -> None:
    """Start playback in background; use when you want to return a reply to the user immediately."""
    _player.play_async(query)


def add_to_queue(query: str) -> None:
    _player.add_to_queue(query)


def skip() -> bool:
    return _player.skip()


def pause() -> bool:
    return _player.pause()


def resume() -> bool:
    return _player.resume()


def stop() -> bool:
    return _player.stop()


def volume_up() -> int:
    """Increase volume; restarts current track if playing. Returns new volume 0–100."""
    return _player.volume_up()


def volume_down() -> int:
    """Decrease volume; restarts current track if playing. Returns new volume 0–100."""
    return _player.volume_down()


def get_volume() -> int:
    return _player.get_volume()


def is_playing() -> bool:
    """True if music is currently playing."""
    return _player.is_playing()


def duck() -> None:
    """Lower music volume so speech is heard (mpv only)."""
    _player.duck()


def unduck() -> None:
    """Restore music volume after ducking."""
    _player.unduck()
