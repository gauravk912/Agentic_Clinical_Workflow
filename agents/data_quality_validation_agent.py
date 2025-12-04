import os
import json 
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from agents.base import BaseAgent
from agents.tools import anomaly_detection_tool
from core.anomaly_detection import train_row_anomaly_model
from core.state import (
    PipelineState,
    ColumnProfile,
    DatasetProfile,
    DataQualityDecision,
)
from config.settings import (
    DQ_MISSINGNESS_WARN_THRESHOLD,
    DQ_MISSINGNESS_FAIL_THRESHOLD,
    DQ_OUTLIER_WARN_THRESHOLD,
)

class DataQualityValidationAgent(BaseAgent):
    """
    Agent that:
      1. Analyzes the DataFrame and builds a DatasetProfile.
      2. Detects potential issues (missingness, type problems, anomalies).
      3. Optionally uses an ML-based anomaly model (IsolationForest) for row-level anomalies.
      4. Calls an LLM to turn that into a human-readable report + PASS/WARN/FAIL Flag.
      5. Saves a markdown data quality report to disk.
    """

    def __init__(self):
        super().__init__(name="data_quality_validation_agent")

    def run(self, state: PipelineState) -> PipelineState:
        df: Optional[pd.DataFrame] = state.get("df") 
        if df is None:
            raise ValueError("DataFrame not found in state. Did load_data_node run?")

        # Target column is - DEATH_EVENT
        target_column = "DEATH_EVENT" if "DEATH_EVENT" in df.columns else None

        # 1. Build dataset profile + issues (Rule-based layer)
        profile, issues = self._build_profile_and_issues(df, target_column)
        
        # DEBUG: inspect numeric_stats for a few columns
        # print("\n[DEBUG] Sample ColumnProfile numeric_stats:")
        # for cp in profile.columns[:5]:  # first 5 columns
        #     print(f"  - {cp.name}: {cp.numeric_stats}")
        # print()
        
        # DEBUG: dump full DatasetProfile JSON to file
        # run_id = state.get("run_id", "debug_run")
        # os.makedirs("outputs/debug", exist_ok=True)
        # debug_profile_path = os.path.join(
        #     "outputs", "debug", f"dq_profile_{run_id}.json"
        # )
        # with open(debug_profile_path, "w", encoding="utf-8") as f:
        #     f.write(profile.model_dump_json(indent=2))
        # print(f"[DEBUG] Saved DatasetProfile JSON to: {debug_profile_path}")

        # 2. ML-based row anomaly detection (IsolationForest)
        model, anomaly_summary = train_row_anomaly_model(
            df, target_column=target_column
        )
        state["row_anomaly_summary"] = anomaly_summary
        frac = anomaly_summary.get("anomaly_fraction", 0.0)
        if frac > 0.1:
            issues.append(
                f"Row-level anomaly detector (IsolationForest) flagged about {frac:.1%} of rows as anomalous."
            )

        # 3. Call LLM to generate final decision (status + report)
        decision = self._decide_with_llm(profile, issues)

        # 4. Write everything back into the shared state
        state["dq_profile"] = profile # Structured pydantic object
        state["dq_decision"] = decision #pydantic dataQualityDecision
        state["validation_report"] = decision.summary #Markdown summary
        state["validation_status"] = decision.status # Pass, warn, Fail

        # 5. Save markdown DQ report to disk
        run_id = state.get("run_id", "unknown_run") 
        os.makedirs("outputs/reports", exist_ok=True)
        report_path = os.path.join(
            "outputs", "reports", f"data_quality_report_{run_id}.md"
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Data Quality Report\n\n")
            f.write(decision.summary)
            if getattr(decision, "issues", None):
                f.write("\n\n## Issues\n")
                for issue in decision.issues:
                    f.write(f"- {issue}\n")

        print(f"[DataQualityValidationAgent] Saved DQ report to: {report_path}")
        print(f"[DataQualityValidationAgent] Status = {decision.status}")

        return state

    # --------- Internal helpers ---------

    def _build_profile_and_issues(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Tuple[DatasetProfile, List[str]]:
    # helper is responsible for DatasetProfile and list of issues found by schema-like info
        n_rows, n_columns = df.shape
        n_duplicate_rows = int(df.duplicated().sum())

        issues: List[str] = []
        column_profiles: List[ColumnProfile] = []

        # columns that should never be negative
        non_negative_cols = {
            "age",
            "creatinine_phosphokinase",
            "ejection_fraction",
            "platelets",
            "serum_creatinine",
            "serum_sodium",
            "time",
        }

        for col_name in df.columns:
            series = df[col_name]
            missing_count = int(series.isna().sum())
            missing_pct = float(missing_count / n_rows) if n_rows > 0 else 0.0
            n_unique = int(series.nunique(dropna=True))
            inferred_type = self._infer_type(series)

            # Initializing numeric stats container
            numeric_stats: Optional[Dict[str, float]] = None
            n_outliers = 0
            n_impossible_values = 0

            if inferred_type == "numeric":
                numeric_series = pd.to_numeric(series, errors="coerce")
                valid = numeric_series.dropna()
                if not valid.empty:
                    mean = float(valid.mean())
                    std = float(valid.std())
                    min_val = float(valid.min())
                    max_val = float(valid.max())
                    q1 = float(valid.quantile(0.25))
                    q3 = float(valid.quantile(0.75))

                    numeric_stats = {
                        "mean": mean,
                        "std": std,
                        "min": min_val,
                        "max": max_val,
                        "q1": q1,
                        "q3": q3,
                    }

                    # Outliers via IQR
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    n_outliers = int(((valid < lower) | (valid > upper)).sum())

                    # Impossible values
                    if col_name in non_negative_cols:
                        n_impossible_values = int((valid < 0).sum())
                        if n_impossible_values > 0:
                            issues.append(
                                f"Column '{col_name}' has {n_impossible_values} negative values but should be non-negative."
                            )

                    # Flag many outliers
                    if n_rows > 0:
                        outlier_pct = n_outliers / n_rows
                        if outlier_pct > DQ_OUTLIER_WARN_THRESHOLD:
                            issues.append(
                                f"Column '{col_name}' has {n_outliers} outliers (~{outlier_pct:.1%} of rows)."
                            )

            # Missingness issues
            if missing_pct > DQ_MISSINGNESS_FAIL_THRESHOLD:
                issues.append(
                    f"Column '{col_name}' has {missing_pct:.1%} missing values (above FAIL threshold)."
                )
            elif missing_pct > DQ_MISSINGNESS_WARN_THRESHOLD:
                issues.append(
                    f"Column '{col_name}' has {missing_pct:.1%} missing values (above WARN threshold)."
                )

            # Type issues
            if inferred_type == "mixed":
                issues.append(
                    f"Column '{col_name}' has mixed types (both numeric-like and string-like values)."
                )

            col_profile = ColumnProfile(
                name=col_name,
                inferred_type=inferred_type,
                missing_count=missing_count,
                missing_pct=missing_pct,
                n_unique=n_unique,
                numeric_stats=numeric_stats,
                # mean = mean,
                # std=std,
                # min = min,
                # max = max,
                # q1=q1,
                # q3 = q3,
                n_outliers=n_outliers,
                n_impossible_values=n_impossible_values,
            )
            column_profiles.append(col_profile)

        # Target column checks
        if target_column is not None and target_column in df.columns:
            target_series = df[target_column]
            if target_series.isna().any():
                issues.append(
                    f"Target column '{target_column}' has missing values, which is problematic for supervised learning."
                )
            counts = target_series.value_counts(dropna=False).to_dict()

            if len(counts) <= 1:
                issues.append(
                    f"Target column '{target_column}' has only one class present."
                )
            else:
                # Simple imbalance check
                total = sum(counts.values())
                max_class_pct = max(counts.values()) / total
                if max_class_pct > 0.9:
                    issues.append(
                        f"Target column '{target_column}' is highly imbalanced (largest class ~{max_class_pct:.1%})."
                    )

        # Dataset-level issues
        if n_duplicate_rows > 0:
            dup_pct = n_duplicate_rows / n_rows if n_rows > 0 else 0.0
            issues.append(
                f"Dataset has {n_duplicate_rows} duplicate rows (~{dup_pct:.1%} of rows)."
            )

        # IMPORTANT: pass n_duplicate_rows because DatasetProfile requires it
        profile = DatasetProfile(
            n_rows=n_rows,
            n_columns=n_columns,
            n_duplicate_rows=n_duplicate_rows,
            columns=column_profiles,
        )

        return profile, issues

    def _infer_type(self, series: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        coerced = pd.to_numeric(series, errors="coerce")
        n_non_null = series.notna().sum()
        n_numeric_like = coerced.notna().sum()

        if n_non_null == 0:
            return "categorical"

        frac_numeric_like = n_numeric_like / n_non_null

        if frac_numeric_like > 0.9:
            return "numeric"
        elif frac_numeric_like > 0.1:
            return "mixed"
        else:
            return "categorical"
    
    def _decide_with_llm(
        self,
        profile: DatasetProfile,
        issues: List[str],
    ) -> DataQualityDecision:

        structured_llm = self.llm.with_structured_output(DataQualityDecision)

        issues_text = "\n".join(f"- {issue}" for issue in issues) or "None detected by rules."

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert data quality analyst for machine learning pipelines. "
                "You will be given a dataset profile and a list of automatically detected issues. "
                "Your job is to: (1) summarize overall data quality, (2) highlight key risks, "
                "and (3) assign a status: PASS, WARN, or FAIL.\n\n"
                "Interpretation of statuses:\n"
                "- PASS = dataset is generally usable, only minor issues.\n"
                "- WARN = dataset is usable but has moderate issues that should be addressed.\n"
                "- FAIL = dataset has severe issues and should not proceed to modeling without fixes.\n\n"
                "You must return a DataQualityDecision object where the 'summary' field is a "
                "well-formatted markdown report with the following structure:\n\n"
                "# Data Quality Report\n\n"
                '**Status:** `<PASS/WARN/FAIL>` (first line after the title)\n\n'
                "## Overview\n"
                "- 1–3 sentences describing dataset size, basic completeness, and overall quality.\n\n"
                "## Column-level checks\n"
                "- Bullet points describing missing values, data types, outliers, and impossible values.\n"
                "- When relevant, mention thresholds (e.g., 'above FAIL threshold of 50% missing').\n\n"
                "## Row-level anomalies\n"
                "- Summarize any row-level anomaly detection in detail (e.g., IsolationForest) results, also mention the types of anomaly and where including approximate percentage of anomalous rows.\n\n"
                "## Recommendation\n"
                "- Clear recommendation on whether to proceed with modeling and what fixes or mitigations are needed for each column-level and row-level anomalies.\n"
                "- Also clearly mention/suggest how can the data types, outliers, impossible values and anomalies be corrected or handled\n\n"
                "## Issues\n"
                "- Bullet list of the most important issues detected.\n\n"
                "Keep the language clear and accessible to both DS/ML engineers and clinical stakeholders."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                "Here is the dataset profile (JSON):\n"
                f"{profile.model_dump_json(indent=2)}\n\n"
                "Here are issues detected by simple rules and the anomaly detection model:\n"
                f"{issues_text}\n\n"
                "Please analyze this information and produce a DataQualityDecision."
            ),
        }

        decision: DataQualityDecision = structured_llm.invoke([system_msg, user_msg])
        return decision
