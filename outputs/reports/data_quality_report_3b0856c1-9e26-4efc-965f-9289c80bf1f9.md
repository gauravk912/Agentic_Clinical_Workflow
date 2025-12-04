# Data Quality Report

# Data Quality Report

**Status:** `WARN`

## Overview
- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness.
- Overall, the data quality is reasonable, but there are moderate concerns related to outliers and row-level anomalies that require attention.

## Column-level checks
- No missing values detected in any column, which supports data completeness.
- All columns are numeric, with binary variables (e.g., anaemia, diabetes, sex) showing expected distributions.
- Numeric columns such as `creatinine_phosphokinase`, `platelets`, and `serum_creatinine` have notable outliers (29, 21, and 29 respectively), which could impact model performance.
- `ejection_fraction` and `serum_sodium` also have a small number of outliers (2 and 4 respectively).
- No impossible values were detected in any column.
- Numeric statistics indicate reasonable ranges for clinical variables, but the presence of high maximum values and large standard deviations in some columns suggests potential extreme values.

## Row-level anomalies
- IsolationForest anomaly detection flagged approximately 10% of rows as anomalous.
- These anomalies may correspond to unusual patient profiles or data entry errors and could bias model training if not addressed.

## Recommendation
- Proceed with modeling but address the identified issues to improve robustness.
- Investigate and potentially cap or transform outliers in `creatinine_phosphokinase`, `platelets`, `serum_creatinine`, `ejection_fraction`, and `serum_sodium`.
- Consider domain knowledge to decide if outliers represent valid extreme cases or errors.
- Review the 10% anomalous rows flagged by IsolationForest; consider exclusion or separate modeling if anomalies represent distinct subpopulations.
- Data types are appropriate; no changes needed.
- Use robust scaling or outlier-resistant algorithms to mitigate impact of extreme values.

## Issues
- Presence of outliers in several numeric columns (notably `creatinine_phosphokinase`, `platelets`, `serum_creatinine`).
- Approximately 10% of rows flagged as anomalous by IsolationForest.
- Large variability in some numeric features that may affect model stability.

## Issues
- Outliers detected in creatinine_phosphokinase (29), platelets (21), serum_creatinine (29), ejection_fraction (2), and serum_sodium (4) columns.
- IsolationForest flagged about 10% of rows as anomalous, indicating potential data anomalies or rare cases.
