# Data Quality Report

# Data Quality Report

## Overview
The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, which is a positive indicator of data completeness. However, several numeric columns exhibit outliers, and the row-level anomaly detection flagged approximately 10% of the rows as anomalous. These factors suggest moderate data quality concerns that should be addressed before modeling.

## Column-level checks
- **Missingness:** No missing values detected in any column.
- **Data Types:** All columns are numeric as expected, with binary variables (e.g., anaemia, diabetes, sex) showing 2 unique values.
- **Outliers:** Notable outliers were detected in several columns:
  - `creatinine_phosphokinase`: 29 outliers
  - `ejection_fraction`: 2 outliers
  - `platelets`: 21 outliers
  - `serum_creatinine`: 29 outliers
  - `serum_sodium`: 4 outliers
  These outliers may represent extreme but valid clinical values or data entry errors and warrant further investigation.

## Row-level anomalies
- Approximately 10% of rows were flagged as anomalous by the IsolationForest algorithm. This relatively high anomaly rate indicates potential data quality issues such as unusual patient profiles or measurement errors.

## Recommendation
Proceed with caution. The dataset is generally usable but contains moderate risks due to multiple outliers and a significant proportion of anomalous rows. It is recommended to:
- Investigate and validate outlier values to determine if they are clinically plausible or errors.
- Review anomalous rows for potential data quality issues or special cases.
- Consider data cleaning or transformation strategies to mitigate the impact of anomalies on modeling.
Addressing these issues will improve model reliability and clinical interpretability.

## Issues
- Row-level anomaly detector (IsolationForest) flagged about 10.0% of rows as anomalous.
- Multiple columns contain outliers: creatinine_phosphokinase (29), ejection_fraction (2), platelets (21), serum_creatinine (29), serum_sodium (4).
- No missing values or duplicates detected.
