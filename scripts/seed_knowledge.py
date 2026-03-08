"""
Seed the robot knowledge base (FTS) with TRES content.
Run from project root: python scripts/seed_knowledge.py
Re-running replaces all knowledge entries.
"""

import sys
from pathlib import Path

# Run from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory.store import MemoryStore, DB_PATH


# Chunks optimized for FTS: key terms in each so bot finds them with few tokens.
# One topic per chunk; concise for fast retrieval and low prompt size.
KNOWLEDGE_CHUNKS = [
    # --- Philosophy & tone ---
    ("tres_philosophy", "The ordinary needs to be challenged. Most people end up doing meaningless jobs. We have real problems to solve—becoming a consultant? Come on."),
    ("tres_philosophy", "Building from scratch has become stupidly easy. There is no real reason not to build, only excuses. Most people find a stupid excuse when confronted; good for you."),
    ("tres_philosophy", "Surrounded by cool, kind people in love with tech and solving real-world problems. Exploring, experimenting, having fun, building freely with friends and a lot of support."),
    ("tres_philosophy", "We are university students who love making things happen for the love and fun of it."),
    ("tres_philosophy", "Remember: less smart thinking and more stupid doing."),
    # --- Benefits ---
    ("tres_benefits", "We have coffee, a 3D printer, and other benefits for members."),
    # --- Summer Founders Program (SFP) ---
    ("tres_sfp", "Summer Founders Program SFP is coming again this year. Just promise you'll give all you've got to the life of building."),
    ("tres_sfp", "Last year's Summer Founders Program was a success. Long days, failed ideas and companies, but also successes—the brightest being Kova Labs. We look for unique and ambitious ideas. Ready to innovate and build? This program is for you."),
    ("tres_sfp", "SFP 2026 runs May 11 to August 5: validation period in Tampere, then about 15 days in San Francisco, then build phase. It concludes with Demo Day where you pitch your vision to investors."),
    ("tres_sfp", "Build with Doers: for builders, innovators, curious minds who want to create something of real value. You don't need an existing company—you need the right mindset and the drive to keep grinding. We provide resources and people; only you can make it count."),
    ("tres_sfp", "What SFP provides: validation period in Tampere (accommodation, resources for validation); over two weeks in San Francisco (accommodation and flights); building phase—place to build, learn pitching, connect with investors; a bubble where things get done."),
    # --- Robolabs ---
    ("tres_robolabs", "Robolabs is a one-weekend workshop where you can build anything robotics-related. We are bringing robotics to everyone."),
    ("tres_robolabs", "Robolabs how it works: planning day February 10 (ideas, choose parts, order); building weekend March 6–8 (plans into action). We provide resources and help; you bring curiosity and motivation. Limited spots—we notify as soon as possible."),
]


def main() -> None:
    store = MemoryStore(DB_PATH)
    deleted = store.clear_knowledge()
    print(f"[seed_knowledge] Cleared {deleted} existing knowledge rows.")
    for source, content in KNOWLEDGE_CHUNKS:
        store.add_knowledge(source, content)
    print(f"[seed_knowledge] Inserted {len(KNOWLEDGE_CHUNKS)} chunks (sources: tres_philosophy, tres_benefits, tres_sfp, tres_robolabs).")
    store.close()


if __name__ == "__main__":
    main()
