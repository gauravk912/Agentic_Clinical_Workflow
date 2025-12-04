import json
import pandas as pd
from langchain_core.tools import tool

from core.anomaly_detection import train_row_anomaly_model


@tool
def anomaly_detection_tool(csv_path: str, target_column: str = "DEATH_EVENT") -> str:
    df = pd.read_csv(csv_path)
    model, summary = train_row_anomaly_model(df, target_column=target_column)
    return json.dumps(summary, indent=2)




    # Run row-level anomaly detection (IsolationForest) on the given CSV file
    # and return a JSON summary with anomaly statistics.

    # Args:
    #     csv_path: Path to the CSV dataset.
    #     target_column: Name of the target column (excluded from features).

    # Returns:
    #     JSON string with keys:
    #       - n_samples
    #       - n_features
    #       - n_anomalies
    #       - anomaly_fraction
    #       - anomaly_score_min/max/mean