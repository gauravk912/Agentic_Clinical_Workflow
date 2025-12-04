# Data Quality Report

# Data Quality Report

## Overview
This dataset contains 309 rows and 13 columns, primarily numeric features related to clinical measurements. While most columns have no missing values, there are significant concerns including a high proportion of missing data in a key feature, presence of impossible negative values in some clinical measurements, duplicate rows, and a notable fraction of anomalous rows detected. These issues collectively pose a serious risk to model reliability and validity.

## Column-level checks
- **Missingness:**
  - `serum_creatinine` has 51.5% missing values, which is above the acceptable threshold and severely limits its usability.
  - Target column `DEATH_EVENT` has 2.6% missing values, problematic for supervised learning.
- **Impossible values:**
  - `ejection_fraction` contains 2 negative values, which are clinically impossible.
  - `serum_sodium` contains 2 negative values, also impossible.
- **Duplicates:**
  - There are 10 duplicate rows (~3.2%), which may bias model training.
- **Outliers:**
  - Several columns (`creatinine_phosphokinase`, `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium`) have outliers, which should be further investigated.

## Row-level anomalies
- IsolationForest flagged approximately 10% of rows as anomalous, indicating potential data quality or distribution issues that could affect model performance.

## Recommendation
Given the severity of missing data in `serum_creatinine`, missing target values, presence of impossible negative values, duplicates, and a high anomaly rate, this dataset should **not** proceed to modeling without substantial cleaning and validation. Key fixes include imputing or removing missing values, correcting or removing impossible values, deduplicating rows, and investigating anomalies. Only after these issues are addressed should modeling be considered.

---

## Issues
- Column 'ejection_fraction' has 2 negative values but should be non-negative.
- Column 'serum_creatinine' has 51.5% missing values (above FAIL threshold).
- Column 'serum_sodium' has 2 negative values but should be non-negative.
- Target column 'DEATH_EVENT' has missing values, which is problematic for supervised learning.
- Dataset has 10 duplicate rows (~3.2% of rows).
- Row-level anomaly detector (IsolationForest) flagged about 10.0% of rows as anomalous.
