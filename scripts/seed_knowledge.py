"""
Load knowledge from data/knowledge/*.txt into the database.
Run from project root: python scripts/seed_knowledge.py
Re-running replaces all knowledge entries.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, DB_PATH


def main() -> None:
    store = MemoryStore(DB_PATH)
    count = store.load_knowledge_from_text_dir()
    print(f"[seed_knowledge] Loaded {count} chunks from data/knowledge/*.txt")
    store.close()


if __name__ == "__main__":
    main()
