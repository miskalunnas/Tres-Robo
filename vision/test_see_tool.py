"""Test see-tool pipeline: camera + encode + (mock or real) Vision API.

Run from project root:
  python -m vision.test_see_tool              # uses mock API, real camera
  python -m vision.test_see_tool --real-api   # real OpenAI call (needs OPENAI_API_KEY)
"""
import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Test see (vision) tool")
    parser.add_argument("--real-api", action="store_true", help="Use real OpenAI API (default: mock)")
    args = parser.parse_args()

    if args.real_api:
        import os
        from openai import OpenAI
        if not os.environ.get("OPENAI_API_KEY"):
            print("Set OPENAI_API_KEY for --real-api", file=sys.stderr)
            sys.exit(1)
        client = OpenAI()
    else:
        # Mock client: capture_and_describe will get real camera + encode, then this returns fixed text
        class MockChoice:
            message = type("Msg", (), {"content": "Test: näen kameran kuvan. (mock)"})()

        class MockClient:
            class chat:
                class completions:
                    @staticmethod
                    def create(*_, **__):
                        return type("Resp", (), {"choices": [MockChoice()]})()

        client = MockClient()

    from vision.scene import capture_and_describe

    question = "Kuvaile lyhyesti mitä näet."
    print(f"[test_see_tool] Calling capture_and_describe(question={question!r}, client={'real' if args.real_api else 'mock'})...")
    result = capture_and_describe(question, client)
    print(f"[test_see_tool] Result: {result!r}")
    if not result or "En saanut" in result or "puuttuu" in result or "epäonnistui" in result or "En pystynyt" in result:
        print("[test_see_tool] FAIL: got error message.", file=sys.stderr)
        sys.exit(1)
    print("[test_see_tool] OK: see tool returned a description.")


if __name__ == "__main__":
    main()
