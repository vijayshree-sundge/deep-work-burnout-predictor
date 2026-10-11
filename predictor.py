"""Glue code: load models, run predictions, build the result dictionary."""
import joblib
import pandas as pd

from focus_curve import find_slumps_and_peaks, group_ranges, hourly_focus
from train_model import MODEL_PATH, train

MAX_DEEP_WORK_HOURS = 6.0  # realistic ceiling of deep work in one day


def load_models():
    if not MODEL_PATH.exists():
        train()
    return joblib.load(MODEL_PATH)


def burnout_label(p: float) -> str:
    if p < 35:
        return "Low"
    if p < 65:
        return "Moderate"
    return "High"


def predict(profile: dict, bundle: dict | None = None) -> dict:
    bundle = bundle or load_models()
    X = pd.DataFrame([profile])[bundle["features"]]

    productivity = float(bundle["models"]["productivity_score"].predict(X)[0])
    burnout = float(bundle["models"]["burnout_probability"].predict(X)[0])
    productivity = min(max(productivity, 0), 100)
    burnout = min(max(burnout, 0), 100)

    curve = hourly_focus(
        productivity,
        profile["sleep_hours"],
        profile["caffeine_cups"],
        profile["morning_mood"],
    )
    slumps, peaks = find_slumps_and_peaks(curve)

    return {
        "productivity": productivity,
        "burnout": burnout,
        "burnout_label": burnout_label(burnout),
        "deep_work_hours": round(productivity / 100 * MAX_DEEP_WORK_HOURS, 1),
        "curve": curve,
        "slump_hours": slumps,
        "peak_hours": peaks,
        "slump_ranges": group_ranges(slumps),
        "peak_ranges": group_ranges(peaks),
    }
