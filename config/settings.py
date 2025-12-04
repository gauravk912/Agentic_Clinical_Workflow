import os
from dotenv import load_dotenv 


load_dotenv()

# ---- API keys/model config ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4.1-mini")

# ---- Data quality thresholds (Hyperparameters) ---- 
DQ_MISSINGNESS_WARN_THRESHOLD = float(os.getenv("DQ_MISSINGNESS_WARN_THRESHOLD", "0.1"))
DQ_MISSINGNESS_FAIL_THRESHOLD = float(os.getenv("DQ_MISSINGNESS_FAIL_THRESHOLD", "0.3"))
DQ_OUTLIER_WARN_THRESHOLD = float(os.getenv("DQ_OUTLIER_WARN_THRESHOLD", "0.1"))

# ---- Dataset config ----
DATASET_VARIANT = os.getenv("DATASET_VARIANT", "clean")  # clean or dirty
DATA_DIR = os.getenv("DATA_DIR", "data")
