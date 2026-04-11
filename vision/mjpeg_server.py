"""Lightweight MJPEG HTTP server for live camera preview.

Disabled by default — enable with CAMERA_STREAM=1 in .env.

Usage:
    # Start once at bot startup:
    from vision.mjpeg_server import start as start_stream
    start_stream()

    # Push frames from anywhere (no-op if not started):
    from vision import mjpeg_server
    mjpeg_server.push_frame(bgr_numpy_frame)

Then open on your laptop:
    http://<pi-ip>:8080
"""

import io
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

STREAM_PORT = int(os.environ.get("CAMERA_STREAM_PORT", "8080"))

# ── Internal state ────────────────────────────────────────────────────────────

_latest_jpeg: bytes | None = None
_frame_lock   = threading.Lock()
_frame_event  = threading.Event()
_started      = False


def push_frame(bgr_frame) -> None:
    """Push a BGR numpy frame to the stream. No-op if server is not started."""
    if not _started:
        return
    try:
        import cv2
        ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        global _latest_jpeg
        with _frame_lock:
            _latest_jpeg = buf.tobytes()
        _frame_event.set()
        _frame_event.clear()
    except Exception:
        pass  # never let streaming break the bot


def _get_frame(timeout: float = 1.0) -> bytes | None:
    _frame_event.wait(timeout=timeout)
    with _frame_lock:
        return _latest_jpeg


# ── HTTP handler ──────────────────────────────────────────────────────────────

_INDEX = b"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tres-Robo Camera</title>
  <style>
    body { margin:0; background:#111; display:flex; flex-direction:column;
           align-items:center; justify-content:center; min-height:100vh;
           color:#eee; font-family:sans-serif; }
    img  { max-width:100%; border:2px solid #333; border-radius:4px; }
    h2   { margin-bottom:12px; letter-spacing:1px; }
  </style>
</head>
<body>
  <h2>Tres-Robo \u2014 Live Camera</h2>
  <img src="/stream" alt="Camera stream">
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request noise from the bot's terminal

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_INDEX)))
            self.end_headers()
            self.wfile.write(_INDEX)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    jpeg = _get_frame(timeout=2.0)
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        elif self.path == "/snapshot":
            with _frame_lock:
                jpeg = _latest_jpeg
            if jpeg is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)

        else:
            self.send_response(404)
            self.end_headers()


# ── Public start function ─────────────────────────────────────────────────────

def start(port: int = STREAM_PORT) -> None:
    """Start the MJPEG server in a background daemon thread."""
    global _started
    if _started:
        return
    _started = True

    def _serve():
        server = HTTPServer(("0.0.0.0", port), _Handler)
        print(f"[Stream] MJPEG server started — open http://<pi-ip>:{port} on your laptop")
        server.serve_forever()

    threading.Thread(target=_serve, daemon=True, name="mjpeg-server").start()
