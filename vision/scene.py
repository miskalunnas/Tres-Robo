"""On-demand scene description via GPT-4o-mini Vision.

Called only when the LLM explicitly invokes the 'see' tool.
No camera activity happens during normal turns.
"""

import base64
import sys


def capture_and_describe(question: str, client) -> str:
    """Capture one frame from the camera and ask GPT-4o-mini Vision to answer *question*."""
    from vision.camera import Camera

    try:
        import cv2
    except ImportError:
        print("[Vision] opencv-python not installed. Run: pip install opencv-python-headless", file=sys.stderr)
        return "Kameraohjelmisto puuttuu."

    # Capture one frame — warmup is short since we only need a single snapshot.
    try:
        with Camera(warmup_seconds=0.4) as cam:
            frame = cam.capture()  # BGR numpy array (H, W, 3)
    except Exception as exc:
        print(f"[Vision] Camera capture error: {exc}", file=sys.stderr)
        return "En saanut kuvaa kamerasta."

    # Always save last frame for debugging — inspect at /tmp/vision_debug.jpg
    cv2.imwrite("/tmp/vision_debug.jpg", frame)
    print(f"[Vision] Debug frame saved to /tmp/vision_debug.jpg (shape={frame.shape})")

    # Face recognition — runs in-memory, fast, gracefully skipped if not installed.
    names: list[str] = []
    try:
        from vision.identity_manager import FaceManager
        names = FaceManager.get().recognize_faces(frame)
        if names:
            print(f"[Vision] Recognized: {names}")
        else:
            print("[Vision] No known faces recognized.")
    except Exception as exc:
        print(f"[Vision] Face recognition skipped: {exc}", file=sys.stderr)

    # Encode as JPEG. detail="low" in the API caps at 512 px anyway, so quality 75 is fine.
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    if not ok or buf is None:
        return "Kuvan koodaus epäonnistui."
    b64 = base64.b64encode(buf.tobytes()).decode()

    # Prepend identity hint to question so Vision model has context while answering.
    identity_hint = ("Tunnistetut henkilöt kuvassa: " + ", ".join(names) + ". ") if names else ""
    prompt = identity_hint + question

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",  # ~85 image tokens, fastest + cheapest
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=120,
            timeout=15,
        )
        description = (resp.choices[0].message.content or "").strip()
        # Always prefix recognized names in the returned string so the bot can't miss them.
        if names:
            return "Huoneessa on: " + ", ".join(names) + ". " + description
        return description
    except Exception as exc:
        print(f"[Vision] API error: {exc}", file=sys.stderr)
        if names:
            return "Huoneessa on: " + ", ".join(names) + "."
        return "En pystynyt analysoimaan kuvaa."
