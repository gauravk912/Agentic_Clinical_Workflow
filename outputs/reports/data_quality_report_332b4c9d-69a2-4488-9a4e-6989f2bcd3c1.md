# Data Quality Report

# Data Quality Report

**Status:** WARN

## Overview
- The dataset contains 299 rows and 13 columns, with no missing values or duplicate rows, indicating good completeness.
- Overall, the data quality is reasonable, but there are some concerns regarding outliers in several numeric columns and a notable proportion of row-level anomalies.

## Column-level checks
- All columns have 0% missing values, which is excellent.
- Several numeric columns contain outliers: creatinine_phosphokinase (29 outliers), ejection_fraction (2 outliers), platelets (21 outliers), serum_creatinine (29 outliers), and serum_sodium (4 outliers).
- No impossible values were detected in any column.
- Data types are consistent with expectations (mostly numeric with binary categorical encoded as numeric).

## Row-level anomalies
- The IsolationForest anomaly detector flagged approximately 10% of rows as anomalous.
- These anomalies may indicate unusual patient profiles or data entry inconsistencies that could impact model performance.

## Recommendation
- Proceed with modeling but address the outliers by investigating their causes; consider winsorizing or transforming these values if they are errors or extreme but valid values.
- Review the 10% anomalous rows flagged by the IsolationForest to determine if they represent data errors or rare but valid cases; consider excluding or separately modeling these if appropriate.
- Maintain current data types; no changes needed.
- No missing data handling is required.

## Issues
- Presence of outliers in multiple numeric columns (creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, serum_sodium).
- Approximately 10% of rows flagged as anomalous by IsolationForest.


## Issues
- Outliers detected in creatinine_phosphokinase, ejection_fraction, platelets, serum_creatinine, and serum_sodium columns.
- 10% of rows flagged as anomalous by IsolationForest anomaly detection.
