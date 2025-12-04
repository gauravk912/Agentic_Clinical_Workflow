import os
import numpy as np
import pandas as pd

# 1. Load the clean dataset
clean_path = os.path.join("data", "heart_failure_clean.csv")
df_clean = pd.read_csv(clean_path)

df_dirty = df_clean.copy()

# 2. Introduce missingness in a key numeric feature (serum_creatinine)
rng = np.random.default_rng(seed=42)
mask_sc = rng.choice([True, False], size=len(df_dirty), p=[0.5, 0.5])
df_dirty.loc[mask_sc, "serum_creatinine"] = np.nan

# 3. Introduce missingness in the target column DEATH_EVENT
df_dirty.loc[2:5, "DEATH_EVENT"] = np.nan

# 4. Type inconsistency: write some 'age' values as the string "unknown"
idx_age_str = df_dirty.sample(5, random_state=123).index
df_dirty.loc[idx_age_str, "age"] = "unknown"

# 5. Impossible negative values in non-negative columns
df_dirty.loc[0, "serum_sodium"] = -5
df_dirty.loc[1, "ejection_fraction"] = -10

# 6. Add duplicate rows: duplicate first 10 rows
df_dirty = pd.concat([df_dirty, df_dirty.head(10)], ignore_index=True)

# 7. Save to a new file
os.makedirs("data", exist_ok=True)
dirty_path = os.path.join("data", "heart_failure_dirty.csv")
df_dirty.to_csv(dirty_path, index=False)

print(f"Created dirty dataset at: {dirty_path}")
print(f"Original shape: {df_clean.shape}, Dirty shape: {df_dirty.shape}")
