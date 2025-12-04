# # agents/data_quality_validation_agent.py

# from typing import List, Tuple, Optional, Dict

# import numpy as np
# import pandas as pd
# from core.anomaly_detection import train_row_anomaly_model

# from agents.base import BaseAgent
# from core.state import (
#     PipelineState,
#     ColumnProfile,
#     DatasetProfile,
#     DataQualityDecision,
# )
# from config.settings import (
#     DQ_MISSINGNESS_WARN_THRESHOLD,
#     DQ_MISSINGNESS_FAIL_THRESHOLD,
#     DQ_OUTLIER_WARN_THRESHOLD,
# )


# class DataQualityValidationAgent(BaseAgent):
#     """
#     Agent that:
#       1. Analyzes the DataFrame and builds a DatasetProfile.
#       2. Detects potential issues (missingness, type problems, anomalies).
#       3. Calls an LLM to turn that into a human-readable report + PASS/WARN/FAIL.
#     """

#     def __init__(self):
#         super().__init__(name="data_quality_validation_agent")

#     # --------- Public entrypoint called from LangGraph node ---------

#     def run(self, state: PipelineState) -> PipelineState:
#         df: Optional[pd.DataFrame] = state.get("df")  # type: ignore
#         if df is None:
#             raise ValueError("DataFrame not found in state. Did load_data_node run?")

#         # For this dataset, we know the target column is 'DEATH_EVENT'
#         target_column = "DEATH_EVENT" if "DEATH_EVENT" in df.columns else None

#         # 1. Build dataset profile + raw issues from simple rules
#         profile, issues = self._build_profile_and_issues(df, target_column)

#         # 2. Call LLM to generate final decision (status + report)
#         decision = self._decide_with_llm(profile, issues)

#         # 3. Write everything back into the shared state
#         state["dq_profile"] = profile
#         state["dq_decision"] = decision
#         state["validation_report"] = decision.summary
#         state["validation_status"] = decision.status

        
#         # New update for Summary Agent
        
#         return state

#     # --------- Internal helpers ---------

#     def _build_profile_and_issues(
#         self,
#         df: pd.DataFrame,
#         target_column: Optional[str] = None,
#     ) -> Tuple[DatasetProfile, List[str]]:
#         n_rows, n_columns = df.shape
#         n_duplicate_rows = int(df.duplicated().sum())

#         issues: List[str] = []
#         column_profiles: List[ColumnProfile] = []

#         # Domain-specific: columns that should never be negative
#         non_negative_cols = {
#             "age",
#             "creatinine_phosphokinase",
#             "ejection_fraction",
#             "platelets",
#             "serum_creatinine",
#             "serum_sodium",
#             "time",
#         }

#         for col_name in df.columns:
#             series = df[col_name]
#             missing_count = int(series.isna().sum())
#             missing_pct = float(missing_count / n_rows) if n_rows > 0 else 0.0
#             n_unique = int(series.nunique(dropna=True))

#             # Infer type: numeric / categorical / mixed
#             inferred_type = self._infer_type(series)

#             # Initialize numeric stats
#             mean = std = min_val = max_val = q1 = q3 = None
#             n_outliers = None
#             n_impossible_values = None

#             if inferred_type == "numeric":
#                 numeric_series = pd.to_numeric(series, errors="coerce")
#                 valid = numeric_series.dropna()
#                 if not valid.empty:
#                     mean = float(valid.mean())
#                     std = float(valid.std())
#                     min_val = float(valid.min())
#                     max_val = float(valid.max())
#                     q1 = float(valid.quantile(0.25))
#                     q3 = float(valid.quantile(0.75))

#                     # Outliers via IQR
#                     iqr = q3 - q1
#                     lower = q1 - 1.5 * iqr
#                     upper = q3 + 1.5 * iqr
#                     n_outliers = int(((valid < lower) | (valid > upper)).sum())

#                     # Impossible values: negative where not allowed
#                     if col_name in non_negative_cols:
#                         n_impossible_values = int((valid < 0).sum())
#                         if n_impossible_values > 0:
#                             issues.append(
#                                 f"Column '{col_name}' has {n_impossible_values} negative values but should be non-negative."
#                             )

#                     # Flag many outliers
#                     if n_outliers is not None and n_rows > 0:
#                         outlier_pct = n_outliers / n_rows
#                         if outlier_pct > DQ_OUTLIER_WARN_THRESHOLD:
#                             issues.append(
#                                 f"Column '{col_name}' has {n_outliers} outliers (~{outlier_pct:.1%} of rows)."
#                             )

#             # Missingness issues
#             if missing_pct > DQ_MISSINGNESS_FAIL_THRESHOLD:
#                 issues.append(
#                     f"Column '{col_name}' has {missing_pct:.1%} missing values (above FAIL threshold)."
#                 )
#             elif missing_pct > DQ_MISSINGNESS_WARN_THRESHOLD:
#                 issues.append(
#                     f"Column '{col_name}' has {missing_pct:.1%} missing values (above WARN threshold)."
#                 )

#             # Type issues
#             if inferred_type == "mixed":
#                 issues.append(
#                     f"Column '{col_name}' has mixed types (both numeric-like and string-like values)."
#                 )

#             col_profile = ColumnProfile(
#                 name=col_name,
#                 inferred_type=inferred_type,
#                 missing_count=missing_count,
#                 missing_pct=missing_pct,
#                 n_unique=n_unique,
#                 mean=mean,
#                 std=std,
#                 min=min_val,
#                 max=max_val,
#                 q1=q1,
#                 q3=q3,
#                 n_outliers=n_outliers,
#                 n_impossible_values=n_impossible_values,
#             )
#             column_profiles.append(col_profile)

#         # Target column checks (if known)
#         target_dist: Optional[Dict[str, int]] = None
#         if target_column is not None and target_column in df.columns:
#             target_series = df[target_column]
#             if target_series.isna().any():
#                 issues.append(
#                     f"Target column '{target_column}' has missing values, which is problematic for supervised learning."
#                 )
#             counts = target_series.value_counts(dropna=False).to_dict()
#             target_dist = {str(k): int(v) for k, v in counts.items()}

#             if len(counts) <= 1:
#                 issues.append(
#                     f"Target column '{target_column}' has only one class present."
#                 )
#             else:
#                 # Simple imbalance check
#                 total = sum(counts.values())
#                 max_class_pct = max(counts.values()) / total
#                 if max_class_pct > 0.9:
#                     issues.append(
#                         f"Target column '{target_column}' is highly imbalanced (largest class ~{max_class_pct:.1%})."
#                     )

#         # Dataset-level issues
#         if n_duplicate_rows > 0:
#             dup_pct = n_duplicate_rows / n_rows if n_rows > 0 else 0.0
#             issues.append(
#                 f"Dataset has {n_duplicate_rows} duplicate rows (~{dup_pct:.1%} of rows)."
#             )

#         profile = DatasetProfile(
#             n_rows=n_rows,
#             n_columns=n_columns,
#             n_duplicate_rows=n_duplicate_rows,
#             columns=column_profiles,
#             target_column=target_column,
#             target_class_distribution=target_dist,
#             notes="Profile generated by DataQualityValidationAgent.",
#         )

#         return profile, issues

#     def _infer_type(self, series: pd.Series) -> str:
#         """
#         Infer column type: numeric / categorical / mixed.
#         """
#         # If pandas thinks it's numeric, trust that
#         if pd.api.types.is_numeric_dtype(series):
#             return "numeric"

#         # Otherwise, try to coerce to numeric and see how many succeed
#         coerced = pd.to_numeric(series, errors="coerce")
#         n_non_null = series.notna().sum()
#         n_numeric_like = coerced.notna().sum()

#         if n_non_null == 0:
#             return "categorical"

#         frac_numeric_like = n_numeric_like / n_non_null

#         if frac_numeric_like > 0.9:
#             return "numeric"
#         elif frac_numeric_like > 0.1:
#             return "mixed"
#         else:
#             return "categorical"

#     def _decide_with_llm(
#         self,
#         profile: DatasetProfile,
#         issues: List[str],
#     ) -> DataQualityDecision:
#         """
#         Use the LLM (via LangChain) to turn the dataset profile + raw issues
#         into a structured DataQualityDecision.
#         """
#         # We use LangChain's structured output feature to directly get a Pydantic object.
#         structured_llm = self.llm.with_structured_output(DataQualityDecision)

#         issues_text = "\n".join(f"- {issue}" for issue in issues) or "None detected by rules."

#         system_msg = {
#             "role": "system",
#             "content": (
#                 "You are an expert data quality analyst for machine learning pipelines. "
#                 "You will be given a dataset profile and a list of automatically detected issues. "
#                 "Your job is to: (1) summarize overall data quality, (2) highlight key risks, "
#                 "and (3) assign a status: PASS, WARN, or FAIL.\n\n"
#                 "PASS = dataset is generally usable, only minor issues.\n"
#                 "WARN = dataset is usable but has moderate issues that should be addressed.\n"
#                 "FAIL = dataset has severe issues and should not proceed to modeling without fixes.\n"
#                 "Return your answer as a DataQualityDecision object."
#             ),
#         }

#         user_msg = {
#             "role": "user",
#             "content": (
#                 "Here is the dataset profile (JSON):\n"
#                 f"{profile.model_dump_json(indent=2)}\n\n"
#                 "Here are issues detected by simple rules:\n"
#                 f"{issues_text}\n\n"
#                 "Please analyze this information and produce a DataQualityDecision."
#             ),
#         }

#         decision: DataQualityDecision = structured_llm.invoke([system_msg, user_msg])
#         return decision

