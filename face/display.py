"""face/display.py — Emotional face display for 800x480 HDMI screen.

States are driven automatically by bot activity — the LLM never calls this directly.

  IDLE      logo + cycling startup quote, dim breathing face
  LISTENING session active, waiting for user to speak
  THINKING  processing / waiting for Gemini response
  SPEAKING  playing audio response
  HAPPY     positive reaction (auto-returns to previous state after 2.5 s)
  SAD       negative reaction (auto-returns to previous state after 2.5 s)

Usage:
    from face.display import FaceState, set_state, start_display
    start_display()
    set_state(FaceState.LISTENING)
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Optional

# ── Constants ──────────────────────────────────────────────────────────────────

W, H = 800, 480
FPS = 60
CX, CY = W // 2, H // 2

_BG   = (6, 6, 18)
_CYAN = (0, 210, 255)
_DIM  = (0, 60, 80)
_GOLD = (255, 190, 50)
_BLUE = (60, 80, 255)

_ROOT       = Path(__file__).resolve().parent.parent
_LOGO_PATH  = _ROOT / "face" / "assets" / "logo.png"

_EMOTION_DURATION  = 2.5   # seconds before HAPPY/SAD reverts
_QUOTE_INTERVAL    = 14.0  # seconds each quote is shown
_QUOTE_FADE        = 2.0   # seconds for quote crossfade


# ── Startup quotes ─────────────────────────────────────────────────────────────

_QUOTES = [
    "Move fast and build things.",
    "The best startups change what people think is possible.",
    "Make something people want.",
    "Do things that don't scale.",
    "Stay hungry. Stay foolish.",
    "Ideas are easy. Execution is everything.",
    "Build, measure, learn. Repeat.",
    "If you're not embarrassed by v1, you launched too late.",
    "Fall in love with the problem, not the solution.",
    "The only way to win is to learn faster than anyone else.",
    "Ship it.",
    "Fortune favors the bold.",
    "First solve the problem, then write the code.",
    "A small team with a big idea can change the world.",
    "Done is better than perfect.",
    "Startups don't fail from a lack of ideas — they fail from a lack of execution.",
    "Your most unhappy customers are your greatest source of learning.",
    "The secret to getting ahead is getting started.",
]


# ── State ──────────────────────────────────────────────────────────────────────

class FaceState(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"
    HAPPY     = "happy"
    SAD       = "sad"


# ── Per-state base parameters ──────────────────────────────────────────────────

_PARAMS: dict[FaceState, dict] = {
    FaceState.IDLE:      dict(eye_open=0.20, brow= 0.0, mouth=  0, color=_DIM),
    FaceState.LISTENING: dict(eye_open=0.90, brow= 0.1, mouth=  8, color=_CYAN),
    FaceState.THINKING:  dict(eye_open=0.60, brow= 0.3, mouth=  0, color=_CYAN),
    FaceState.SPEAKING:  dict(eye_open=0.85, brow= 0.1, mouth= 18, color=_CYAN),
    FaceState.HAPPY:     dict(eye_open=0.45, brow= 0.5, mouth= 45, color=_GOLD),
    FaceState.SAD:       dict(eye_open=0.55, brow=-0.4, mouth=-38, color=_BLUE),
}


# ── Singleton ──────────────────────────────────────────────────────────────────

_display: Optional[FaceDisplay] = None


def set_state(state: FaceState) -> None:
    if _display is not None:
        _display.set_state(state)


def start_display() -> None:
    global _display
    _display = FaceDisplay()
    _display.start()


# ── FaceDisplay ────────────────────────────────────────────────────────────────

class FaceDisplay:
    def __init__(self) -> None:
        self._state      = FaceState.IDLE
        self._prev_state = FaceState.IDLE
        self._prev_state_to_restore: Optional[FaceState] = None
        self._emotion_set_at   = 0.0
        self._transition_start = 0.0
        self._transition_dur   = 0.35
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Blink
        self._next_blink = time.monotonic() + 3.0
        self._blink_t    = 0.0
        # Quote cycling
        self._quote_idx     = 0
        self._quote_started = time.monotonic()

    def set_state(self, state: FaceState) -> None:
        with self._lock:
            if state == self._state:
                return
            if state in (FaceState.HAPPY, FaceState.SAD):
                self._prev_state_to_restore = self._state
                self._emotion_set_at = time.monotonic()
            else:
                self._prev_state_to_restore = None
            self._prev_state       = self._state
            self._state            = state
            self._transition_start = time.monotonic()

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="face-display"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    # ── render loop ────────────────────────────────────────────────────────────

    def _run(self) -> None:
        screen = self._open_display()
        if screen is None:
            print("[Face] No display available — face module disabled.", file=sys.stderr)
            return

        import pygame

        pygame.mouse.set_visible(False)
        clock = pygame.time.Clock()

        # Load logo (optional — graceful if missing)
        logo_surf = self._load_logo(pygame)

        # Load font for quotes
        quote_font = self._load_font(pygame, size=22)
        author_font = self._load_font(pygame, size=15)

        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False

            t = time.monotonic()

            with self._lock:
                state       = self._state
                prev_state  = self._prev_state
                blend       = min(1.0, (t - self._transition_start) / self._transition_dur)
                restore     = self._prev_state_to_restore
                emotion_age = t - self._emotion_set_at

            if restore is not None and emotion_age >= _EMOTION_DURATION:
                self.set_state(restore)

            self._tick_blink(t, state)

            # idle_alpha: 1.0 when fully IDLE, 0.0 when fully in any other state
            if state == FaceState.IDLE:
                idle_alpha = blend
            elif prev_state == FaceState.IDLE:
                idle_alpha = 1.0 - blend
            else:
                idle_alpha = 0.0

            screen.fill(_BG)
            self._draw(screen, t, state, prev_state, blend)

            if idle_alpha > 0.01:
                self._draw_idle_overlay(
                    screen, pygame, t, logo_surf, quote_font, author_font, idle_alpha
                )

            pygame.display.flip()
            clock.tick(FPS)

        pygame.quit()

    def _open_display(self):
        try:
            import pygame
        except ImportError:
            return None

        for driver in ("kmsdrm", "fbdev", ""):
            try:
                if driver:
                    os.environ["SDL_VIDEODRIVER"] = driver
                else:
                    os.environ.pop("SDL_VIDEODRIVER", None)
                pygame.init()
                screen = pygame.display.set_mode(
                    (W, H), pygame.FULLSCREEN | pygame.NOFRAME
                )
                pygame.display.set_caption("Tres")
                print(f"[Face] Display opened (driver={driver or 'default'}) {W}x{H}")
                return screen
            except Exception as exc:
                print(f"[Face] SDL driver '{driver or 'default'}' failed: {exc}", file=sys.stderr)
                try:
                    pygame.quit()
                except Exception:
                    pass
        return None

    def _load_logo(self, pygame):
        if not _LOGO_PATH.exists():
            print(f"[Face] Logo not found at {_LOGO_PATH} — skipping.", file=sys.stderr)
            return None
        try:
            surf = pygame.image.load(str(_LOGO_PATH)).convert_alpha()
            # Scale to max 220 px wide, keeping aspect ratio
            ow, oh = surf.get_size()
            scale = min(220 / ow, 100 / oh)
            nw, nh = int(ow * scale), int(oh * scale)
            return pygame.transform.smoothscale(surf, (nw, nh))
        except Exception as exc:
            print(f"[Face] Logo load failed: {exc}", file=sys.stderr)
            return None

    def _load_font(self, pygame, size: int):
        for name in ("dejavusans", "liberationsans", "freesans", "arial", ""):
            try:
                if name:
                    f = pygame.font.SysFont(name, size)
                else:
                    f = pygame.font.Font(None, size + 6)
                return f
            except Exception:
                pass
        return pygame.font.Font(None, size + 6)

    # ── blink ──────────────────────────────────────────────────────────────────

    def _tick_blink(self, t: float, state: FaceState) -> None:
        if state == FaceState.IDLE:
            self._blink_t = 0.0
            return
        if t >= self._next_blink:
            elapsed = t - self._next_blink
            if elapsed < 0.08:
                self._blink_t = elapsed / 0.08
            elif elapsed < 0.18:
                self._blink_t = 1.0 - (elapsed - 0.08) / 0.10
            else:
                self._blink_t = 0.0
                self._next_blink = t + 3.5 + 1.5 * abs(math.sin(t * 0.7))

    # ── lerp ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _lerp_f(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    @staticmethod
    def _lerp_color(a: tuple, b: tuple, t: float) -> tuple:
        return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(len(a)))

    def _blend_params(self, state: FaceState, prev: FaceState, blend: float) -> dict:
        a, b = _PARAMS[prev], _PARAMS[state]
        return {
            "eye_open":  self._lerp_f(a["eye_open"], b["eye_open"], blend),
            "brow":      self._lerp_f(a["brow"],     b["brow"],     blend),
            "mouth":     self._lerp_f(a["mouth"],    b["mouth"],    blend),
            "color":     self._lerp_color(a["color"], b["color"],   blend),
        }

    # ── idle overlay ───────────────────────────────────────────────────────────

    def _draw_idle_overlay(self, screen, pygame, t, logo_surf, quote_font, author_font, alpha):
        """Draw logo + cycling quote on top of the IDLE face."""

        # ── logo ──────────────────────────────────────────────────────────────
        if logo_surf is not None:
            # Gentle pulse brightness with breathing
            pulse = 0.55 + 0.20 * math.sin(t * 0.8)
            a = int(255 * alpha * pulse)
            logo = logo_surf.copy()
            logo.set_alpha(a)
            lw, lh = logo.get_size()
            # Position: horizontally centered, vertically above the face center
            screen.blit(logo, (CX - lw // 2, CY - 155 - lh // 2))

        # ── quote cycling ─────────────────────────────────────────────────────
        age = t - self._quote_started
        if age >= _QUOTE_INTERVAL:
            self._quote_idx = (self._quote_idx + 1) % len(_QUOTES)
            self._quote_started = t
            age = 0.0

        # Fade in for first _QUOTE_FADE seconds, fade out for last _QUOTE_FADE seconds
        if age < _QUOTE_FADE:
            q_alpha = age / _QUOTE_FADE
        elif age > _QUOTE_INTERVAL - _QUOTE_FADE:
            q_alpha = ((_QUOTE_INTERVAL - age) / _QUOTE_FADE)
        else:
            q_alpha = 1.0

        q_alpha = max(0.0, min(1.0, q_alpha)) * alpha

        quote = _QUOTES[self._quote_idx]
        self._draw_centered_text(
            screen, pygame, quote_font,
            text=f'"{quote}"',
            color=_CYAN,
            alpha=q_alpha,
            y=H - 68,
            max_width=680,
        )

    def _draw_centered_text(self, screen, pygame, font, text, color, alpha, y, max_width):
        """Render text centered horizontally at y, wrapping if needed."""
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            test = (current + " " + word).strip()
            if font.size(test)[0] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)

        line_h = font.get_linesize()
        total_h = line_h * len(lines)
        start_y = y - total_h // 2

        for i, line in enumerate(lines):
            surf = font.render(line, True, color)
            surf.set_alpha(int(255 * alpha))
            lw = surf.get_width()
            screen.blit(surf, (CX - lw // 2, start_y + i * line_h))

    # ── main draw ──────────────────────────────────────────────────────────────

    def _draw(self, screen, t, state, prev, blend):
        import pygame

        p = self._blend_params(state, prev, blend)
        eye_open    = float(p["eye_open"])
        brow        = float(p["brow"])
        mouth_curve = float(p["mouth"])
        color       = p["color"]
        eye_shift   = 0

        if state == FaceState.IDLE:
            breathe  = 0.15 + 0.10 * math.sin(t * 0.9)
            eye_open = breathe
            fade     = 0.35 + 0.15 * math.sin(t * 0.6)
            color    = tuple(int(v * fade) for v in _CYAN)

        elif state == FaceState.THINKING:
            eye_shift = int(7 * math.sin(t * 2.2))

        elif state == FaceState.SPEAKING:
            pulse       = abs(math.sin(t * 9.0))
            mouth_curve = 12 + int(32 * pulse)

        if state != FaceState.IDLE and self._blink_t > 0:
            eye_open = float(eye_open) * (1.0 - self._blink_t)

        EYE_W     = 72
        EYE_H     = 56
        EYE_SPACE = 112
        EYE_Y     = CY - 28
        MOUTH_Y   = CY + 85
        MOUTH_W   = 130

        lx = CX - EYE_SPACE + eye_shift
        rx = CX + EYE_SPACE + eye_shift

        self._draw_eye(screen, lx, EYE_Y, EYE_W, EYE_H, eye_open, brow, color)
        self._draw_eye(screen, rx, EYE_Y, EYE_W, EYE_H, eye_open, brow, color)

        if state == FaceState.THINKING:
            self._draw_thinking_dots(screen, CX, MOUTH_Y, t, color)
        else:
            self._draw_mouth(screen, CX, MOUTH_Y, MOUTH_W, mouth_curve, color)

        if state == FaceState.HAPPY and blend > 0.4:
            self._draw_sparkles(screen, t, color)

    # ── drawing primitives ─────────────────────────────────────────────────────

    def _draw_eye(self, surf, cx, cy, w, h, openness, brow, color, line=3):
        import pygame
        eye_h = max(2, int(h * openness))
        if eye_h <= 4:
            pygame.draw.line(surf, color, (cx - w // 2, cy), (cx + w // 2, cy), 3)
        else:
            rect = pygame.Rect(cx - w // 2, cy - eye_h // 2, w, eye_h)
            pygame.draw.ellipse(surf, color, rect, line)
            if openness > 0.35:
                pr = max(3, int(min(w, eye_h) * 0.22))
                pygame.draw.circle(surf, color, (cx, cy), pr)
        brow_base_y = cy - eye_h // 2 - 14
        brow_pts = [
            (cx - w // 2 + 6, brow_base_y - int(brow * 5)),
            (cx,              brow_base_y - int(brow * 12)),
            (cx + w // 2 - 6, brow_base_y - int(brow * 5)),
        ]
        pygame.draw.lines(surf, color, False, brow_pts, 2)

    def _draw_mouth(self, surf, cx, cy, w, curve, color, line=4):
        import pygame
        pts = []
        half = w // 2
        for i in range(41):
            frac = i / 40
            x = cx - half + int(w * frac)
            y = cy - int(curve * math.sin(math.pi * frac))
            pts.append((x, y))
        if len(pts) > 1:
            pygame.draw.lines(surf, color, False, pts, line)

    def _draw_thinking_dots(self, surf, cx, cy, t, color):
        import pygame
        for i in range(3):
            phase = (t * 2.5 + i * 0.45) % (math.pi * 2)
            y_off = int(7 * math.sin(phase))
            alpha = 0.45 + 0.55 * ((math.sin(phase) + 1) / 2)
            c     = tuple(int(v * alpha) for v in color)
            pygame.draw.circle(surf, c, (cx - 26 + i * 26, cy + y_off), 8)

    def _draw_sparkles(self, surf, t, color):
        import pygame
        for i in range(7):
            angle = t * 1.3 + i * (math.pi * 2 / 7)
            r     = 185 + int(18 * math.sin(t * 2.1 + i))
            x     = int(CX + r * math.cos(angle))
            y     = int(CY + r * 0.55 * math.sin(angle))
            size  = max(1, 3 + int(2 * math.sin(t * 3.5 + i)))
            pygame.draw.circle(surf, color, (x, y), size)
