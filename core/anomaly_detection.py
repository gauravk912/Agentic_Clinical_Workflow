import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Dict, Any, Tuple, Optional


def train_row_anomaly_model(
    df: pd.DataFrame,
    target_column: Optional[str] = "DEATH_EVENT",
    contamination: float = 0.1,
    random_state: int = 42,
) -> Tuple[Optional[IsolationForest], Dict[str, Any]]:
    feature_df = df.copy()

    if target_column is not None and target_column in feature_df.columns:
        feature_df = feature_df.drop(columns=[target_column])

    X = feature_df.select_dtypes(include=["number"])

    if X.empty:
        return None, {
            "n_samples": len(df),
            "n_features": 0,
            "n_anomalies": 0,
            "anomaly_fraction": 0.0,
            "note": "No numeric features available for anomaly detection.",
        }

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(X)

    preds = model.predict(X)  # -1 = anomaly, 1 = normal
    scores = model.decision_function(X)  # higher = more normal

    n_samples = len(df)
    n_anomalies = int((preds == -1).sum())
    anomaly_fraction = n_anomalies / n_samples if n_samples > 0 else 0.0

    summary: Dict[str, Any] = {
        "n_samples": n_samples,
        "n_features": int(X.shape[1]),
        "n_anomalies": n_anomalies,
        "anomaly_fraction": float(anomaly_fraction),
        "anomaly_score_min": float(scores.min()),
        "anomaly_score_max": float(scores.max()),
        "anomaly_score_mean": float(scores.mean()),
    }

    return model, summary
