# Data Quality Report

# Data Quality Report

**Status:** `WARN`

## Overview
- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness and uniqueness.
- Overall, the data types are consistent with expectations, and there are no impossible values detected.

## Column-level checks
- All columns have 0% missing values, which is excellent.
- Numeric columns such as `age`, `creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, and `serum_sodium` have some outliers detected:
  - `creatinine_phosphokinase` has 29 outliers (approx. 9.7% of rows).
  - `ejection_fraction` has 2 outliers.
  - `platelets` has 21 outliers.
  - `serum_creatinine` has 29 outliers.
  - `serum_sodium` has 4 outliers.
- No impossible values were found in any column.
- Binary columns (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`, `DEATH_EVENT`) have values strictly 0 or 1, which is appropriate.

## Row-level anomalies
- The IsolationForest anomaly detection model flagged approximately 10% of rows as anomalous.
- These anomalies may correspond to unusual combinations of feature values or extreme values in numeric columns.

## Recommendation
- Proceed with modeling but with caution due to the presence of outliers and row-level anomalies.
- Investigate and consider handling outliers in numeric columns by methods such as winsorization, transformation, or removal depending on domain knowledge.
- Review the anomalous rows flagged by the IsolationForest to understand if they represent data errors or rare but valid cases.
- Ensure that binary columns are treated as categorical variables in modeling.
- No missing data imputation is needed.

## Issues
- Presence of outliers in several numeric columns, notably `creatinine_phosphokinase` and `serum_creatinine`.
- Approximately 10% of rows flagged as anomalous by IsolationForest.


## Issues
- 29 outliers detected in 'creatinine_phosphokinase'
- 29 outliers detected in 'serum_creatinine'
- 21 outliers detected in 'platelets'
- 10% of rows flagged as anomalous by IsolationForest
- 4 outliers detected in 'serum_sodium'
- 2 outliers detected in 'ejection_fraction'
