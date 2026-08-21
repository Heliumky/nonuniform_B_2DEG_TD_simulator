"""Small helpers for human-readable numerical-test output."""


def report_check(name, model, measured, limit, unit=""):
    """Print the measured error and the pass criterion used by a test."""
    suffix = f" {unit}" if unit else ""
    print(
        f"\n[CHECK] {name}\n"
        f"  model:     {model}\n"
        f"  measured:  {measured:.3e}{suffix}\n"
        f"  criterion: measured < {limit:.3e}{suffix}"
    )
