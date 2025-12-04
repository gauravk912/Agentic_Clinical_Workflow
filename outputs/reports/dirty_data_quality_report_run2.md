# Data Quality Report

# Data Quality Report

**Status:** `FAIL`

## Overview

- The dataset contains 309 rows and 13 columns, with a moderate number of duplicates and several critical data quality issues. Overall completeness is compromised by a high proportion of missing values in key columns and some invalid data points.
- 

## Column-level checks

- The 'serum_creatinine' column has 51.5% missing values, exceeding the FAIL threshold and severely impacting data completeness.
- The target column 'DEATH_EVENT' has 2.6% missing values, which is problematic for supervised learning tasks.
- 'ejection_fraction' and 'serum_sodium' columns contain negative values, which are impossible given their clinical context.
- There are 10 duplicate rows (~3.2%), which may bias model training if not addressed.
- Several columns have outliers, notably 'creatinine_phosphokinase' (30 outliers), 'platelets' (20 outliers), and 'serum_creatinine' (14 outliers).

## Row-level anomalies

- An IsolationForest anomaly detection model flagged approximately 10% of rows as anomalous, indicating potential data quality or heterogeneity issues.

## Recommendation

- Due to the high missingness in 'serum_creatinine' and invalid negative values in important clinical features, the dataset is not currently suitable for modeling.
- Immediate remediation should include imputing or collecting missing 'serum_creatinine' data, correcting or removing impossible negative values, and handling duplicates.
- Addressing the missing target values is essential before supervised modeling.
- After these fixes, re-assess anomaly rates and outliers.

## Issues

- 'serum_creatinine' has 51.5% missing values (above FAIL threshold).
- Negative values found in 'ejection_fraction' and 'serum_sodium'.
- Missing values in target column 'DEATH_EVENT'.
- Presence of 10 duplicate rows (~3.2%).
- 10% of rows flagged as anomalous by IsolationForest.
- Multiple columns contain outliers that may require further investigation.

## Issues

- 'serum_creatinine' has 51.5% missing values (above FAIL threshold).
- Negative values found in 'ejection_fraction' and 'serum_sodium'.
- Missing values in target column 'DEATH_EVENT'.
- Presence of 10 duplicate rows (~3.2%).
- 10% of rows flagged as anomalous by IsolationForest.
- Multiple columns contain outliers that may require further investigation.
