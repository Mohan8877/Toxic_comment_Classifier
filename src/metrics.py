from __future__ import annotations

from time import monotonic

_START_TIME = monotonic()


def uptime_seconds() -> float:
    return monotonic() - _START_TIME


def calculate_toxicity_score(categories: dict[str, float]) -> float:
    if not categories:
        return 0.0
    return max(float(value) for value in categories.values())


def determine_moderation_action(score: float) -> str:
    if score < 0.3:
        return "allow"
    if score < 0.75:
        return "flag"
    return "hide_and_flag"
