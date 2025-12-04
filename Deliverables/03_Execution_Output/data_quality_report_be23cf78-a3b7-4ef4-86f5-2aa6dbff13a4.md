# Data Quality Report

**Status:** `FAIL`

## Overview

- The dataset contains 309 rows and 13 columns, with a moderate number of duplicates (10 rows, ~3.2%). Overall completeness is compromised by significant missing data in key columns and some invalid values.

## Column-level checks

- The 'serum_creatinine' column has 51.5% missing values, which exceeds the fail threshold and severely impacts usability.
- The target column 'DEATH_EVENT' has missing values (~2.6%), which is problematic for supervised learning tasks.
- 'ejection_fraction' contains 2 negative values, which are impossible since this metric should be non-negative.
- 'serum_sodium' also contains 2 negative values, which are invalid.
- Several columns have outliers: 'creatinine_phosphokinase' (30 outliers), 'ejection_fraction' (4 outliers), 'platelets' (20 outliers), 'serum_creatinine' (14 outliers), and 'serum_sodium' (10 outliers).

## Row-level anomalies

- The IsolationForest anomaly detector flagged approximately 10% of rows as anomalous, indicating a notable presence of unusual or inconsistent data points.

## Recommendation

- The dataset currently should not proceed to modeling without remediation.
- Impute or otherwise address the 51.5% missing data in 'serum_creatinine'—consider advanced imputation methods or removal if imputation is not feasible.
- Address missing values in the target 'DEATH_EVENT' by either imputing carefully or removing affected rows to avoid bias.
- Correct impossible negative values in 'ejection_fraction' and 'serum_sodium' by verifying data entry or setting them to null for imputation.
- Investigate and handle outliers using domain knowledge; consider winsorization or robust scaling.
- Remove duplicate rows to avoid bias.
- Review and potentially exclude or further investigate rows flagged as anomalies by IsolationForest to improve model robustness.

## Issues

- 51.5% missing values in 'serum_creatinine' (above fail threshold).
- Missing values in target column 'DEATH_EVENT'.
- Impossible negative values in 'ejection_fraction' and 'serum_sodium'.
- Presence of multiple outliers in key numeric columns.
- 10 duplicate rows (~3.2%).
- 10% of rows flagged as anomalous by IsolationForest.

Addressing these issues is critical before any reliable modeling can be performed to ensure data integrity and model performance.

## Issues

- Column 'serum_creatinine' has 51.5% missing values, exceeding fail threshold.
- Target column 'DEATH_EVENT' has missing values, problematic for supervised learning.
- Column 'ejection_fraction' contains 2 negative (impossible) values.
- Column 'serum_sodium' contains 2 negative (impossible) values.
- Dataset contains 10 duplicate rows (~3.2%).
- IsolationForest detected ~10% anomalous rows.
- Multiple columns have outliers that require review and handling.
