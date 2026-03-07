"""Standalone vision test pipeline.

Commands:
    # Enrol a new person (captures N frames, saves embeddings):
    python -m vision.test_vision --enrol "Your Name"

    # Run continuous recognition loop (prints who it sees every second):
    python -m vision.test_vision --run

    # Single snapshot — detect and identify once, then exit:
    python -m vision.test_vision --snapshot

    # List all enrolled people:
    python -m vision.test_vision --list

    # Remove a person by id:
    python -m vision.test_vision --remove <id>
"""
import argparse
import sys
import time

from vision.camera import Camera
from vision.face_db import FaceDB
from vision.face_id import detect_and_encode

ENROL_FRAMES = 5          # how many frames to capture when enrolling
ENROL_FRAME_DELAY = 0.5   # seconds between enrol captures
RUN_INTERVAL = 1.0        # seconds between recognition attempts in --run mode


def cmd_enrol(name: str, db: FaceDB) -> None:
    print(f"Enrolling '{name}' — capturing {ENROL_FRAMES} frames. Look at the camera.")
    embeddings = []
    with Camera() as cam:
        for i in range(ENROL_FRAMES):
            time.sleep(ENROL_FRAME_DELAY)
            frame = cam.capture()
            found = detect_and_encode(frame)
            if not found:
                print(f"  Frame {i + 1}/{ENROL_FRAMES}: no face detected — skipping")
                continue
            if len(found) > 1:
                print(f"  Frame {i + 1}/{ENROL_FRAMES}: {len(found)} faces detected — use only one person")
                continue
            embeddings.append(found[0])
            print(f"  Frame {i + 1}/{ENROL_FRAMES}: face captured")

    if not embeddings:
        print("[ERROR] No usable frames captured. Enrolment failed.")
        sys.exit(1)

    person = db.enrol(name, embeddings)
    print(f"Done! '{person.name}' enrolled with id={person.id} using {len(embeddings)} embedding(s).")


def cmd_snapshot(db: FaceDB) -> None:
    print("Capturing snapshot...")
    with Camera() as cam:
        frame = cam.capture()

    encodings = detect_and_encode(frame)
    if not encodings:
        print("No faces detected in frame.")
        return

    print(f"{len(encodings)} face(s) detected:")
    for i, enc in enumerate(encodings):
        person = db.identify(enc)
        label = person.name if person else "Unknown"
        print(f"  Face {i + 1}: {label}")


def cmd_run(db: FaceDB) -> None:
    print(f"Running continuous recognition (every {RUN_INTERVAL:.1f}s). Ctrl+C to stop.")
    with Camera() as cam:
        try:
            while True:
                frame = cam.capture()
                encodings = detect_and_encode(frame)

                if not encodings:
                    print("[No faces]")
                else:
                    names = []
                    for enc in encodings:
                        person = db.identify(enc)
                        names.append(person.name if person else "Unknown")
                    print(f"[Detected] {', '.join(names)}")

                time.sleep(RUN_INTERVAL)
        except KeyboardInterrupt:
            print("\nStopped.")


def cmd_list(db: FaceDB) -> None:
    persons = db.list_persons()
    if not persons:
        print("No persons enrolled yet.")
        return
    print(f"{len(persons)} enrolled person(s):")
    for p in persons:
        print(f"  id={p.id}  name={p.name}  embeddings={len(p.embeddings)}  enrolled={p.enrolled_at}")


def cmd_remove(person_id: str, db: FaceDB) -> None:
    if db.remove(person_id):
        print(f"Removed person id={person_id}.")
    else:
        print(f"No person found with id={person_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tres-Robo vision test pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--enrol", metavar="NAME", help="Enrol a new person")
    group.add_argument("--snapshot", action="store_true", help="Single-frame recognition")
    group.add_argument("--run", action="store_true", help="Continuous recognition loop")
    group.add_argument("--list", action="store_true", help="List enrolled persons")
    group.add_argument("--remove", metavar="ID", help="Remove a person by id")
    args = parser.parse_args()

    db = FaceDB()

    if args.enrol:
        cmd_enrol(args.enrol, db)
    elif args.snapshot:
        cmd_snapshot(db)
    elif args.run:
        cmd_run(db)
    elif args.list:
        cmd_list(db)
    elif args.remove:
        cmd_remove(args.remove, db)


if __name__ == "__main__":
    main()
