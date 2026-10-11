"""Generate a synthetic student sleep / study dataset.

Real, labelled "productivity" data is hard to find, so for a mini project we
simulate it using relationships that match published sleep-science findings:
more sleep, better sleep quality and a good morning mood raise productivity,
while heavy screen time, lots of caffeine and over-packed study plans lower it
and increase burnout risk.

Run:  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 2000
OUT_PATH = Path("data") / "student_productivity.csv"


def generate(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    sleep_hours = np.clip(rng.normal(6.8, 1.3, n), 3, 10)
    sleep_quality = np.clip(rng.normal(6.0, 2.0, n), 1, 10)
    screen_time = np.clip(rng.normal(6.0, 2.5, n), 0.5, 14)
    caffeine_cups = np.clip(rng.poisson(2.0, n), 0, 8)
    morning_mood = np.clip(rng.normal(6.0, 2.0, n), 1, 10)
    study_hours_planned = np.clip(rng.normal(5.0, 2.0, n), 0, 12)

    productivity = (
        18
        + 4.5 * sleep_hours
        + 2.2 * sleep_quality
        + 2.5 * morning_mood
        - 2.0 * screen_time
        - 0.8 * caffeine_cups
        - 0.6 * study_hours_planned
        + rng.normal(0, 4, n)
    )

    burnout = (
        38
        - 4.0 * sleep_hours
        - 2.0 * sleep_quality
        - 2.5 * morning_mood
        + 2.5 * screen_time
        + 1.5 * caffeine_cups
        + 2.0 * study_hours_planned
        + 28  # shift so typical values land mid-range
        + rng.normal(0, 5, n)
    )

    df = pd.DataFrame(
        {
            "sleep_hours": sleep_hours.round(1),
            "sleep_quality": sleep_quality.round(1),
            "screen_time_hours": screen_time.round(1),
            "caffeine_cups": caffeine_cups.astype(int),
            "morning_mood": morning_mood.round(1),
            "study_hours_planned": study_hours_planned.round(1),
            "productivity_score": np.clip(productivity, 0, 100).round(1),
            "burnout_probability": np.clip(burnout, 0, 100).round(1),
        }
    )
    return df


if __name__ == "__main__":
    OUT_PATH.parent.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(data)} rows to {OUT_PATH}")
    print(data.describe().round(2))
