"""Train two Linear Regression models and save them.

  * productivity model -> Productivity Score (0-100)
  * burnout model      -> Burnout Probability (0-100 %)

Run:  python train_model.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from generate_data import OUT_PATH, generate

FEATURES = [
    "sleep_hours",
    "sleep_quality",
    "screen_time_hours",
    "caffeine_cups",
    "morning_mood",
    "study_hours_planned",
]
MODEL_PATH = Path("models") / "productivity_models.joblib"
METRICS_PATH = Path("models") / "metrics.json"


def _fit_and_score(X_train, X_test, y_train, y_test):
    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 3),
    }
    return model, metrics


def train() -> dict:
    if not OUT_PATH.exists():
        OUT_PATH.parent.mkdir(exist_ok=True)
        generate().to_csv(OUT_PATH, index=False)

    df = pd.read_csv(OUT_PATH)
    X = df[FEATURES]

    results = {}
    models = {}
    for target in ["productivity_score", "burnout_probability"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, df[target], test_size=0.2, random_state=42
        )
        model, metrics = _fit_and_score(X_train, X_test, y_train, y_test)
        models[target] = model
        results[target] = {
            **metrics,
            "coefficients": {
                f: round(float(c), 3) for f, c in zip(FEATURES, model.coef_)
            },
            "intercept": round(float(model.intercept_), 3),
        }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"models": models, "features": FEATURES}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    out = train()
    for target, m in out.items():
        print(f"\n{target}: R2={m['r2']}  MAE={m['mae']}")
        for feat, coef in m["coefficients"].items():
            print(f"   {feat:22s} {coef:+.3f}")
    print(f"\nSaved model to {MODEL_PATH}")
