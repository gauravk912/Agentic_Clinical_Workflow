# Data Quality Report

# Data Quality Report

**Status:** `WARN`

## Overview
- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness.
- Overall, the data types are consistent and numeric, suitable for modeling.

## Column-level checks
- No missing values detected in any columns.
- No impossible values found.
- Several columns have outliers: `creatinine_phosphokinase` (29 outliers), `platelets` (21 outliers), `serum_creatinine` (29 outliers), `ejection_fraction` (2 outliers), and `serum_sodium` (4 outliers).
- Outliers may represent extreme but valid clinical measurements; however, their presence should be reviewed.

## Row-level anomalies
- IsolationForest anomaly detection flagged approximately 10% of rows as anomalous.
- These anomalies may indicate unusual patient profiles or data quality issues.

## Recommendation
- Proceed with modeling but address the outliers by either capping, transformation, or domain-informed filtering to reduce their impact.
- Investigate the anomalous rows flagged by IsolationForest to determine if they represent true rare cases or data errors.
- Consider robust modeling techniques that can handle outliers and anomalies.
- Maintain current data types as they are appropriate; no corrections needed.

## Issues
- Presence of multiple outliers in key numeric columns.
- 10% of rows flagged as anomalous by IsolationForest.
- No missing or impossible values, but outliers and anomalies pose moderate risks to model performance.

## Issues
- Multiple columns have outliers: creatinine_phosphokinase (29), platelets (21), serum_creatinine (29), ejection_fraction (2), serum_sodium (4).
- IsolationForest detected about 10% of rows as anomalous.
- No missing or impossible values detected.
