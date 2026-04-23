"""Mock lead-capture API integration."""

from __future__ import annotations


def mock_lead_capture(name: str, email: str, platform: str) -> bool:
    """Capture a qualified lead through a mocked external API call."""
    print(f"\n[LEAD CAPTURED] Name: {name}, Email: {email}, Platform: {platform}\n")
    return True