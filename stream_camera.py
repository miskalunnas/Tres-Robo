"""MJPEG camera streaming server for Tres-Robo.

Streams live camera footage over HTTP so you can view it in a browser
on your laptop.  Run this standalone — NOT while the main bot is running,
since picamera2 only allows one process to own the camera at a time.

Usage on the Pi:
    python stream_camera.py

Then open on your laptop:
    http://<pi-ip-address>:8080

The stream URL itself (for VLC or other players):
    http://<pi-ip-address>:8080/stream

Find the Pi's IP:
    hostname -I
"""

import io
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

HOST = "0.0.0.0"
PORT = int(os.environ.get("STREAM_PORT", "8080"))
JPEG_QUALITY = 70   # 1–95; lower = smaller/faster, higher = sharper
TARGET_FPS   = 10


# ── Shared frame buffer ───────────────────────────────────────────────────────

class FrameBuffer:
    def __init__(self):
        self._frame: bytes | None = None
        self._lock  = threading.Lock()
        self._event = threading.Event()

    def put(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self._frame = jpeg_bytes
        self._event.set()
        self._event.clear()

    def get_latest(self) -> bytes | None:
        with self._lock:
            return self._frame

    def wait_new(self, timeout: float = 1.0) -> bytes | None:
        self._event.wait(timeout=timeout)
        return self.get_latest()


_buf = FrameBuffer()


# ── Camera capture thread ─────────────────────────────────────────────────────

def _capture_loop() -> None:
    try:
        import cv2
        from vision.camera import Camera
    except ImportError as exc:
        print(f"[Stream] Import error: {exc}", file=sys.stderr)
        return

    interval = 1.0 / TARGET_FPS
    print(f"[Stream] Opening camera...")
    try:
        with Camera(warmup_seconds=1.0) as cam:
            print(f"[Stream] Camera ready ({cam._backend})")
            while True:
                t0 = time.monotonic()
                try:
                    frame = cam.capture()
                    ok, buf = cv2.imencode(
                        ".jpg", frame,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    if ok:
                        _buf.put(buf.tobytes())
                except Exception as exc:
                    print(f"[Stream] Capture error: {exc}", file=sys.stderr)

                elapsed = time.monotonic() - t0
                sleep_for = interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except Exception as exc:
        print(f"[Stream] Camera error: {exc}", file=sys.stderr)


# ── HTTP handler ──────────────────────────────────────────────────────────────

_INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Tres-Robo Camera</title>
  <style>
    body {{ margin: 0; background: #111; display: flex;
            flex-direction: column; align-items: center; justify-content: center;
            min-height: 100vh; color: #eee; font-family: sans-serif; }}
    img  {{ max-width: 100%; border: 2px solid #333; border-radius: 4px; }}
    h2   {{ margin-bottom: 12px; letter-spacing: 1px; }}
  </style>
</head>
<body>
  <h2>Tres-Robo — Live Camera</h2>
  <img src="/stream" alt="Camera stream">
</body>
</html>
"""


class StreamHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request access log noise

    def do_GET(self):
        if self.path == "/":
            body = _INDEX_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=frame",
            )
            self.end_headers()
            try:
                while True:
                    frame = _buf.wait_new(timeout=2.0)
                    if frame is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"\r\n" + frame + b"\r\n"
                    )
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client disconnected

        elif self.path == "/snapshot":
            frame = _buf.get_latest()
            if frame is None:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

        else:
            self.send_response(404)
            self.end_headers()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t = threading.Thread(target=_capture_loop, daemon=True, name="capture")
    t.start()

    server = HTTPServer((HOST, PORT), StreamHandler)
    print(f"[Stream] Serving at http://0.0.0.0:{PORT}")
    print(f"[Stream] Open on your laptop: http://<pi-ip>:{PORT}")
    print(f"[Stream] Single snapshot:     http://<pi-ip>:{PORT}/snapshot")
    print(f"[Stream] Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[Stream] Stopped.")
