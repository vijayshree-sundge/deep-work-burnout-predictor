"""Hour-by-hour focus curve and slump detection.

The curve starts from a typical circadian rhythm (morning peak, post-lunch dip,
early-evening bounce, night decline) and is then shaped by the user's inputs:
  * less sleep      -> deeper afternoon dip, lower overall level
  * heavy caffeine  -> extra mid/late-afternoon crash
  * low mood        -> weaker morning peak
The overall level is scaled by the predicted productivity score.
"""
import numpy as np

DAY_START, DAY_END = 6, 22  # 6 AM .. 10 PM


def fmt_hour(h: int) -> str:
    return f"{h % 12 or 12}:00 {'AM' if h % 24 < 12 else 'PM'}"


def hourly_focus(
    productivity: float,
    sleep_hours: float,
    caffeine_cups: int,
    morning_mood: float,
) -> dict[int, float]:
    hours = np.arange(DAY_START, DAY_END + 1)

    sleep_debt = max(0.0, 7.5 - sleep_hours)
    morning_peak = 0.30 + 0.02 * (morning_mood - 5)
    dip_depth = 0.28 + 0.05 * sleep_debt
    crash = 0.06 * max(0, caffeine_cups - 2)

    g = lambda h, c, w: np.exp(-(((h - c) / w) ** 2))
    base = (
        0.55
        + morning_peak * g(hours, 10, 2.2)
        + 0.18 * g(hours, 17.5, 2.0)
        - dip_depth * g(hours, 14, 1.3)
        - crash * g(hours, 16, 1.2)
        - 0.25 * np.clip((hours - 20) / 2, 0, 1)
    )
    level = 0.45 + 0.55 * (productivity / 100)
    scores = np.clip(base * level, 0, 1) * 100
    return {int(h): round(float(s), 1) for h, s in zip(hours, scores)}


def find_slumps_and_peaks(curve: dict[int, float]):
    """Return (slump_hours, peak_hours) inside the 8 AM - 9 PM working window."""
    window = {h: s for h, s in curve.items() if 8 <= h <= 21}
    mean = float(np.mean(list(window.values())))
    slumps = [h for h, s in window.items() if s < mean * 0.92]
    peaks = [h for h, s in window.items() if s > mean * 1.08]
    return slumps, peaks


def group_ranges(hours: list[int]) -> list[str]:
    """[13,14,15,19] -> ['1:00 PM - 3:00 PM', '7:00 PM']"""
    if not hours:
        return []
    hours = sorted(hours)
    ranges, start, prev = [], hours[0], hours[0]
    for h in hours[1:]:
        if h == prev + 1:
            prev = h
            continue
        ranges.append((start, prev))
        start = prev = h
    ranges.append((start, prev))
    return [
        fmt_hour(a) if a == b else f"{fmt_hour(a)} - {fmt_hour(b)}"
        for a, b in ranges
    ]
