# Data Quality Report

# Data Quality Report

**Status:** `WARN`

## Overview
- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness.
- Overall, the data types are numeric and consistent across columns.

## Column-level checks
- No missing values detected in any column.
- No impossible values found in any column.
- Several columns have outliers detected: 29 outliers in `creatinine_phosphokinase`, 29 in `serum_creatinine`, 21 in `platelets`, 4 in `serum_sodium`, and 2 in `ejection_fraction`.
- Binary columns (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`, `DEATH_EVENT`) have expected unique values (2) and no outliers.

## Row-level anomalies
- The IsolationForest anomaly detection model flagged approximately 10% of rows as anomalous.
- These anomalies may indicate unusual patient profiles or data entry inconsistencies.

## Recommendation
- Proceed with modeling but address the outliers in numeric columns by investigating their clinical plausibility or applying appropriate transformations or capping.
- Review the 10% anomalous rows flagged by IsolationForest to determine if they represent valid rare cases or data errors; consider excluding or separately modeling these rows.
- Maintain current data types as numeric; no changes needed.
- Implement robust scaling or outlier handling methods to mitigate the impact of outliers on model performance.

## Issues
- Presence of multiple outliers in key numeric columns.
- 10% of rows flagged as anomalous by IsolationForest, indicating potential data quality or heterogeneity concerns.

## Issues
- Multiple columns have outliers: creatinine_phosphokinase (29), serum_creatinine (29), platelets (21), serum_sodium (4), ejection_fraction (2).
- IsolationForest flagged about 10% of rows as anomalous, which may affect model performance if not addressed.
