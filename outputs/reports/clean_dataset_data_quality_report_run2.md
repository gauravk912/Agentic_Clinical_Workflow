# Data Quality Report


**Status:** `WARN`

## Overview

- The dataset contains 299 rows and 13 columns with no missing values or duplicate rows, indicating good completeness.
- Overall data quality is reasonable, but some columns exhibit notable outliers and a moderate proportion of anomalous rows.

## Column-level checks

- No missing values detected in any column.
- Numeric columns such as `creatinine_phosphokinase`, `platelets`, and `serum_creatinine` have a significant number of outliers (29, 21, and 29 respectively).
- `ejection_fraction` and `serum_sodium` also have a few outliers (2 and 4 respectively).
- No impossible values were found in any column.
- Several binary categorical columns (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`, `DEATH_EVENT`) have expected unique values of 2.

## Row-level anomalies

- IsolationForest anomaly detection flagged approximately 10% of rows as anomalous, which is a moderate level of anomalies that could impact model performance.

## Recommendation

- Proceed with modeling but apply careful outlier treatment and consider investigating the anomalous rows further.
- Potentially use robust modeling techniques or data transformations to mitigate the impact of outliers.
- Review the anomalous rows to understand if they represent data errors or rare but valid cases.

## Issues

- Moderate number of outliers in key numeric columns (`creatinine_phosphokinase`, `platelets`, `serum_creatinine`).
- Approximately 10% of rows flagged as anomalous by IsolationForest.
- No missing or impossible values, but outliers and anomalies warrant caution before modeling.

## Issues

- Moderate number of outliers in numeric columns (e.g., creatinine_phosphokinase, platelets, serum_creatinine).
- Approximately 10% of rows flagged as anomalous by IsolationForest.
- No missing or impossible values detected, but outliers and anomalies present moderate risk.
