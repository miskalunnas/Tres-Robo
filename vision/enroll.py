"""CLI tool for enrolling people into the face recognition database.

Usage
-----
# Enroll from an existing image:
python vision/enroll.py --name "Miro" --image /tmp/miro.jpg

# Capture N frames from the camera (default 5) and enroll from the best one:
python vision/enroll.py --name "Miro" --capture

# List enrolled persons:
python vision/enroll.py --list

# Remove a person:
python vision/enroll.py --remove "Miro"
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `vision.*` imports work regardless of
# how/where this script is invoked.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _enroll_from_image(name: str, image_path: str) -> None:
    from vision.identity_manager import FaceManager
    ok = FaceManager.get().register_face(name, image_path)
    if ok:
        print(f"Enrolled '{name}' successfully.")
    else:
        print(f"Failed to enroll '{name}' — no face found in image.", file=sys.stderr)
        sys.exit(1)


def _enroll_from_camera(name: str, n_frames: int) -> None:
    from vision.camera import Camera
    from vision.identity_manager import FaceManager
    from vision import face_id
    import cv2

    print(f"Capturing {n_frames} frame(s) for '{name}'... (look at the camera)")
    manager = FaceManager.get()
    saved = 0

    with Camera(warmup_seconds=1.0) as cam:
        for i in range(n_frames):
            time.sleep(0.5)
            frame = cam.capture()
            encodings = face_id.detect_and_encode(frame)
            if encodings:
                # Save frame to /tmp for inspection, then register
                path = f"/tmp/enroll_{name.replace(' ', '_')}_{i}.jpg"
                cv2.imwrite(path, frame)
                ok = manager.register_face(name, path)
                if ok:
                    saved += 1
                    print(f"  Frame {i+1}/{n_frames}: face captured.")
                else:
                    print(f"  Frame {i+1}/{n_frames}: enrollment failed.")
            else:
                print(f"  Frame {i+1}/{n_frames}: no face detected, skipping.")

    print(f"Done. Saved {saved}/{n_frames} encoding(s) for '{name}'.")


def _list_enrolled() -> None:
    from vision.identity_manager import FaceManager
    names = FaceManager.get().list_enrolled()
    if names:
        print("Enrolled persons:")
        for name in names:
            print(f"  - {name}")
    else:
        print("No persons enrolled yet.")


def _remove(name: str) -> None:
    from vision.face_db import FaceDB
    db = FaceDB()
    persons = [p for p in db.list_persons() if p.name.lower() == name.lower()]
    if not persons:
        print(f"Person '{name}' not found.", file=sys.stderr)
        sys.exit(1)
    for p in persons:
        db.remove(p.id)
        print(f"Removed '{p.name}' (id={p.id}).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Face enrollment tool")
    parser.add_argument("--name", help="Person's display name")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--capture", action="store_true", help="Capture from camera")
    parser.add_argument("--frames", type=int, default=5, help="Frames to capture (default 5)")
    parser.add_argument("--list", action="store_true", help="List enrolled persons")
    parser.add_argument("--remove", metavar="NAME", help="Remove a person by name")
    args = parser.parse_args()

    if args.list:
        _list_enrolled()
    elif args.remove:
        _remove(args.remove)
    elif args.capture:
        if not args.name:
            parser.error("--name required with --capture")
        _enroll_from_camera(args.name, args.frames)
    elif args.image:
        if not args.name:
            parser.error("--name required with --image")
        _enroll_from_image(args.name, args.image)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
