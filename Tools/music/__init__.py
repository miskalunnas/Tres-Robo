"""Music playback via yt-dlp and external player (ffplay / mpv).

Provides queue, skip, pause/resume and stop via a singleton MusicPlayer.
"""

import os
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque


class MusicPlayer:
    """Manages a playback queue and the current player subprocess."""

    def __init__(self) -> None:
        self._queue: deque[str] = deque()
        self._process: subprocess.Popen | None = None
        self._paused: bool = False
        self._stopped: bool = False
        self._lock = threading.Lock()
        self._watcher: threading.Thread | None = None
        self._player_cmd: str | None = None
        self._detect_player()

    # ------------------------------------------------------------------
    # Player detection
    # ------------------------------------------------------------------

    def _detect_player(self) -> None:
        for cmd in ("ffplay", "mpv"):
            if shutil.which(cmd):
                self._player_cmd = cmd
                return
        self._player_cmd = None

    # ------------------------------------------------------------------
    # URL resolution (yt-dlp)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_url(query: str) -> str | None:
        """Use yt-dlp to turn a search query into a direct audio URL."""
        try:
            result = subprocess.run(
                [
                    sys.executable, "-m", "yt_dlp",
                    "-g", "-f", "bestaudio/best",
                    "--no-playlist",
                    "--default-search", "ytsearch",
                    f"ytsearch1:{query}",
                ],
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

    def _start_player(self, url: str) -> subprocess.Popen | None:
        if not self._player_cmd:
            print("[Music] No player found. Install ffmpeg (ffplay) or mpv.", file=sys.stderr)
            return None

        if self._player_cmd == "ffplay":
            args = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", url]
        else:
            args = ["mpv", "--no-video", "--really-quiet", url]

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
            return subprocess.Popen(args, **kwargs)
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
            self._process = self._start_player(url)
            self._paused = False

        if self._process is None:
            return False

        print(f"[Music] Playing: {query!r}")
        self._start_watcher()
        return True

    def add_to_queue(self, query: str) -> None:
        """Append a query to the playback queue."""
        query = (query or "").strip()
        if not query:
            print("[Music] No query to add.")
            return
        with self._lock:
            self._queue.append(query)
        print(f"[Music] Added to queue: {query!r}  (queue length: {len(self._queue)})")

    def skip(self) -> None:
        """Skip the current track and play the next from the queue."""
        with self._lock:
            if self._process is None:
                print("[Music] Nothing playing to skip.")
                return
            self._stopped = False
        print("[Music] Skipping...")
        self._kill_current()

    def pause(self) -> None:
        """Pause the current player process."""
        with self._lock:
            if self._process is None or self._paused:
                print("[Music] Nothing to pause.")
                return
            self._paused = True
            pid = self._process.pid

        if os.name == "nt":
            self._suspend_windows(pid)
        else:
            os.kill(pid, signal.SIGSTOP)
        print("[Music] Paused.")

    def resume(self) -> None:
        """Resume the paused player process."""
        with self._lock:
            if self._process is None or not self._paused:
                print("[Music] Nothing to resume.")
                return
            self._paused = False
            pid = self._process.pid

        if os.name == "nt":
            self._resume_windows(pid)
        else:
            os.kill(pid, signal.SIGCONT)
        print("[Music] Resumed.")

    def stop(self) -> None:
        """Stop playback and clear the queue."""
        with self._lock:
            self._queue.clear()
            self._stopped = True
        self._kill_current()
        print("[Music] Stopped and queue cleared.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _kill_current(self) -> None:
        with self._lock:
            proc = self._process
            if proc is None:
                return
            self._paused = False

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


def add_to_queue(query: str) -> None:
    _player.add_to_queue(query)


def skip() -> None:
    _player.skip()


def pause() -> None:
    _player.pause()


def resume() -> None:
    _player.resume()


def stop() -> None:
    _player.stop()
