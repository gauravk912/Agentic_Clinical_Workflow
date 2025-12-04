import json
import pandas as pd
from langchain_core.tools import tool

from core.anomaly_detection import train_row_anomaly_model


@tool
def anomaly_detection_tool(csv_path: str, target_column: str = "DEATH_EVENT") -> str:
    """
    Run anomaly detection on a CSV file using IQR and Isolation Forest.
    
    Args:
        csv_path: Path to the input CSV file.
        target_column: Name of the target column used to separate outcome from features.
    
    Returns:
        A string (e.g., JSON or file path) summarizing detected anomalies and their counts.
    """
    df = pd.read_csv(csv_path)
    model, summary = train_row_anomaly_model(df, target_column=target_column)
    return json.dumps(summary, indent=2)