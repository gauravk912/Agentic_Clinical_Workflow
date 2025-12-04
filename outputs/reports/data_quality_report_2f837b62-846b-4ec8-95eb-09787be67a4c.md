# Data Quality Report

# Data Quality Report

**Status:** `WARN`

## Overview
- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness.
- Overall data quality is reasonable, but there are several columns with notable outliers and a moderate proportion of anomalous rows detected.

## Column-level checks
- No missing values detected in any column.
- All columns are numeric, with binary variables correctly having two unique values.
- Outliers were detected in several numeric columns: 29 outliers in `creatinine_phosphokinase`, 21 in `platelets`, 29 in `serum_creatinine`, 4 in `serum_sodium`, and 2 in `ejection_fraction`.
- No impossible values were found in any column.
- Numeric statistics show reasonable ranges for clinical variables, but the presence of outliers suggests some extreme values that may affect modeling.

## Row-level anomalies
- IsolationForest anomaly detection flagged approximately 10% of rows as anomalous.
- These anomalies may correspond to unusual combinations of feature values or extreme outliers.

## Recommendation
- Proceed with modeling but address the outliers and anomalies to improve model robustness.
- For columns with outliers (`creatinine_phosphokinase`, `platelets`, `serum_creatinine`, `serum_sodium`, `ejection_fraction`), consider applying transformations, winsorization, or robust scaling.
- Investigate the anomalous rows flagged by IsolationForest to determine if they represent data errors or rare but valid cases; consider excluding or separately modeling these if appropriate.
- Maintain data types as numeric but ensure binary variables are treated as categorical or boolean in modeling.

## Issues
- Presence of outliers in 5 numeric columns.
- Approximately 10% of rows flagged as anomalous by IsolationForest.
- No missing or impossible values, but outliers and anomalies pose moderate risks to model performance.

## Issues
- Outliers detected in creatinine_phosphokinase, platelets, serum_creatinine, serum_sodium, and ejection_fraction columns.
- Approximately 10% of rows flagged as anomalous by IsolationForest.
- No missing or impossible values detected, but outliers and anomalies require attention.
