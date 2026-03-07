"""Motor/servo control (stub)."""


def execute(action: str | dict) -> None:
    """Execute a motor action. Stub: no-op."""
    if isinstance(action, dict):
        action = action.get("action", "")
    if action:
        print(f"[Motors] (stub) would execute: {action}")
