"""Background knowledge curator.

Picks up facts saved at runtime (source='conversation', processed=0) and
integrates them into the appropriate data/knowledge/*.txt files.

When Ollama is running locally it uses a small LLM for classification and
clean-up. If Ollama is unreachable it falls back to keyword-based file
selection and saves the fact as-is.

Usage
-----
# Fire-and-forget from main code:
    from memory.curator import curate_pending
    curate_pending()          # returns immediately; runs in daemon thread

# One-shot from CLI:
    python -m memory.curator
    python -m memory.curator --model qwen2.5:1.5b
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
from pathlib import Path

# Resolved at import time so the module works whether imported or run directly.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

KNOWLEDGE_DIR = _ROOT / "data" / "knowledge"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("CURATOR_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = 10  # seconds

# Map from keyword → target file stem (order matters — first match wins)
_KEYWORD_FILE_MAP: list[tuple[list[str], str]] = [
    (["tres", "sfp", "fuksi", "superfuksi", "hallitus", "pöhinä", "reaktori", "newton"], "tres"),
    (["robolabs", "lab", "laboratorio", "office", "toimisto", "kitchen", "keittiö"], "robolabs"),
    (["lauri", "jooel", "netta", "miska", "oliver", "arttu", "jani", "olli",
      "miro", "diar", "hilma", "ida", "karti", "isäntä", "emäntä"], "tres_people"),
    (["bot", "botin", "robot", "persona", "nimi", "name", "luonne"], "bot_persona"),
    (["komento", "command", "käsky", "ohjaus", "tool", "työkalu"], "robot_commands"),
]
_DEFAULT_FILE = "robolabs"

# Minimum word overlap to consider a fact a near-duplicate of existing content
_DUPLICATE_OVERLAP_THRESHOLD = 0.6


def _classify_file(fact: str) -> str:
    """Return the target file stem for a fact using keyword matching."""
    lower = fact.lower()
    for keywords, stem in _KEYWORD_FILE_MAP:
        if any(kw in lower for kw in keywords):
            return stem
    return _DEFAULT_FILE


def _is_near_duplicate(fact: str, existing_content: str) -> bool:
    """True if the fact is too similar to content already in the file."""
    fact_words = set(re.findall(r"[a-zäöå0-9]+", fact.lower()))
    if len(fact_words) < 3:
        return False
    for para in existing_content.split("\n\n"):
        para_words = set(re.findall(r"[a-zäöå0-9]+", para.lower()))
        if not para_words:
            continue
        overlap = len(fact_words & para_words) / len(fact_words)
        if overlap >= _DUPLICATE_OVERLAP_THRESHOLD:
            return True
    return False


def _call_ollama(fact: str, file_stems: list[str], existing_sample: str) -> dict | None:
    """Ask Ollama to classify and clean the fact. Returns parsed JSON or None on failure."""
    try:
        import urllib.request
    except ImportError:
        return None

    file_list = ", ".join(file_stems)
    prompt = (
        f"You are a knowledge curator for a robot assistant. "
        f"A new fact was learned during a conversation. Your job:\n"
        f"1. Choose which knowledge file it belongs to (one of: {file_list})\n"
        f"2. Clean it into a clear, standalone sentence (preserve original language)\n"
        f"3. Decide if it should be saved or skipped (skip if trivial/temporary/duplicate)\n\n"
        f"New fact: {fact}\n\n"
        f"Sample of existing content (first 500 chars):\n{existing_sample[:500]}\n\n"
        f'Respond with JSON only, no extra text:\n'
        f'{{"target_file": "<stem>", "cleaned_fact": "<sentence>", "skip": false, "reason": "<short>"}}'
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }).encode()

    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
            body = json.loads(resp.read())
        raw = body.get("response", "")
        return json.loads(raw)
    except Exception as exc:
        print(f"[Curator] Ollama call failed: {exc}", file=sys.stderr)
        return None


def _append_to_file(file_path: Path, fact: str) -> None:
    """Append a fact as a new paragraph to the knowledge file."""
    from datetime import date
    note = f"\n\n# Learned {date.today().isoformat()}\n{fact}"
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(note)


def _process_one(store, row_id: int, fact: str) -> None:
    """Process a single unprocessed fact."""
    available_stems = [f.stem for f in sorted(KNOWLEDGE_DIR.glob("*.txt"))]

    # --- keyword-based fallback classification ---
    target_stem = _classify_file(fact)

    # Read existing file content for duplicate check and Ollama context
    target_path = KNOWLEDGE_DIR / f"{target_stem}.txt"
    existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""

    # --- try Ollama for better classification + cleaning ---
    ollama_result = _call_ollama(fact, available_stems, existing)
    if ollama_result:
        if ollama_result.get("skip"):
            reason = ollama_result.get("reason", "")
            print(f"[Curator] Skipped (LLM: {reason}): {fact[:80]}")
            store.mark_knowledge_processed(row_id)
            return
        llm_stem = ollama_result.get("target_file", "").strip()
        if llm_stem in available_stems:
            target_stem = llm_stem
        cleaned = ollama_result.get("cleaned_fact", "").strip() or fact
        # Re-read file if target changed
        target_path = KNOWLEDGE_DIR / f"{target_stem}.txt"
        existing = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    else:
        cleaned = fact

    # --- deduplication ---
    if existing and _is_near_duplicate(cleaned, existing):
        print(f"[Curator] Duplicate, skipping: {cleaned[:80]}")
        store.mark_knowledge_processed(row_id)
        return

    # --- write to file ---
    _append_to_file(target_path, cleaned)
    print(f"[Curator] Appended to {target_stem}.txt: {cleaned[:80]}")

    # --- re-index that file in the DB ---
    try:
        store.reload_knowledge_source(target_stem, target_path)
    except Exception as exc:
        print(f"[Curator] Reload failed: {exc}", file=sys.stderr)

    store.mark_knowledge_processed(row_id)


def process_pending() -> int:
    """Process all unprocessed conversation facts. Returns count processed."""
    from memory.store import MemoryStore
    store = MemoryStore()
    rows = store.list_unprocessed_knowledge("conversation")
    if not rows:
        return 0
    print(f"[Curator] Processing {len(rows)} pending fact(s)...")
    processed = 0
    for row_id, content in rows:
        try:
            _process_one(store, row_id, content)
            processed += 1
        except Exception as exc:
            print(f"[Curator] Error on row {row_id}: {exc}", file=sys.stderr)
    return processed


def curate_pending() -> None:
    """Fire-and-forget: process pending facts in a daemon background thread."""
    t = threading.Thread(target=process_pending, daemon=True, name="knowledge-curator")
    t.start()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Knowledge curator")
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    args = parser.parse_args()
    OLLAMA_MODEL = args.model  # type: ignore[assignment]
    n = process_pending()
    print(f"[Curator] Done. Processed {n} fact(s).")
