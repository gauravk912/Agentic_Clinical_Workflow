# Data Quality Report

# Data Quality Report

**Status:** `WARN`

## Overview
- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness.
- Overall, the data types are consistent and numeric, with no impossible values detected.
- However, there are several columns with notable outliers and a moderate proportion of rows flagged as anomalous.

## Column-level checks
- No missing values or impossible values detected across all columns.
- Outliers detected in several numeric columns:
  - `creatinine_phosphokinase`: 29 outliers (~9.7% of rows), with values ranging up to 7861, which is substantially higher than the third quartile (582).
  - `platelets`: 21 outliers (~7% of rows), with maximum values up to 850,000, which is high compared to the interquartile range.
  - `serum_creatinine`: 29 outliers (~9.7% of rows), with max value 9.4, well above the third quartile (1.4).
  - `serum_sodium`: 4 outliers (~1.3% of rows), with minimum value 113, which may be clinically low.
  - `ejection_fraction`: 2 outliers (~0.7% of rows).
- All categorical-like columns (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`, `DEATH_EVENT`) are numeric with two unique values and no anomalies.

## Row-level anomalies
- IsolationForest anomaly detection flagged approximately 10% of rows as anomalous.
- These anomalies likely correspond to rows with extreme outlier values in key clinical measurements such as `creatinine_phosphokinase`, `serum_creatinine`, and `platelets`.

## Recommendation
- Proceed with modeling but with caution due to the presence of outliers and anomalous rows.
- Investigate outlier values to determine if they are data entry errors or true clinical extremes:
  - Consider winsorizing or capping extreme values in `creatinine_phosphokinase`, `serum_creatinine`, and `platelets`.
  - Alternatively, apply robust scaling or transformation methods to reduce outlier impact.
- For row-level anomalies, consider:
  - Further clinical review to assess if anomalous rows represent rare but valid cases.
  - Potential exclusion or separate modeling if anomalies distort model performance.
- Maintain current data types as numeric; no changes needed.

## Issues
- Presence of multiple columns with moderate numbers of outliers.
- Approximately 10% of rows flagged as anomalous by IsolationForest.
- Potential impact of outliers and anomalies on model robustness and interpretability.

## Issues
- 29 outliers detected in 'creatinine_phosphokinase' (~9.7% of rows)
- 21 outliers detected in 'platelets' (~7% of rows)
- 29 outliers detected in 'serum_creatinine' (~9.7% of rows)
- 4 outliers detected in 'serum_sodium' (~1.3% of rows)
- 2 outliers detected in 'ejection_fraction' (~0.7% of rows)
- IsolationForest flagged approximately 10% of rows as anomalous
