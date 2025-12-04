import joblib
import os
from .state import PipelineState
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import shap


def train_model_and_explain(
    df: pd.DataFrame,
    target_column: str = "DEATH_EVENT",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, float], List[Dict[str, Any]], Any]:

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    # 1. Split into features (X) and label (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 2. Train/test split (stratified so label distribution stays similar)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_names = X.columns.tolist()

    # 3. Build a simple pipeline: StandardScaler + LogisticRegression
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 4. Train model
    model.fit(X_train, y_train)

    # 5. Evaluate on test set
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics: Dict[str, float] = {
        "auc": float(roc_auc_score(y_test, y_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
    }

    # 6. Compute SHAP values for global feature importance
    scaler = model.named_steps["scaler"]
    clf = model.named_steps["clf"]

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    explainer = shap.LinearExplainer(clf, X_train_scaled)
    shap_values = explainer(X_test_scaled)  # SHAP Explanation object

    # shap_values.values has shape (n_samples, n_features)
    values = shap_values.values
    # Compute mean absolute SHAP for each feature
    mean_abs_shap = np.abs(values).mean(axis=0)

    # Rank features by importance
    sorted_indices = np.argsort(mean_abs_shap)[::-1]

    top_features: List[Dict[str, Any]] = []
    for idx in sorted_indices:
        top_features.append(
            {
                "name": feature_names[idx],
                "importance": float(mean_abs_shap[idx]),
            }
        )

    # os.makedirs("outputs/models", exist_ok=True)
    # joblib.dump(model, f"outputs/models/model_{run_id}.joblib")

    return model, metrics, top_features, shap_values



# Train a simple Logistic Regression model on the dataset and compute:
    #   - performance metrics on a held-out test set
    #   - global feature importance using SHAP
    #   - the SHAP explanation object (optional, for deeper inspection)

# Returns:
    #   model: trained sklearn Pipeline (StandardScaler + LogisticRegression)
    #   metrics: dict with keys like 'auc', 'accuracy', 'f1'
    #   top_features: list of {name, importance} sorted by importance desc
    #   shap_explanation: SHAP Explanation object for the test set