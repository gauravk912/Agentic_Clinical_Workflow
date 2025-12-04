from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field

class ColumnProfile(BaseModel):
    name: str # Column name in the DF
    inferred_type: Literal['numeric','categorical','mixed'] # Any one of it
    missing_count: int # How many values are missing
    missing_pct: float # Proportion of missing values
    n_unique: int  # Number of distinct non-null Values
    
    # Numeric Stats
    # mean: Optional[float] = None
    # std: Optional[float] = None
    # min: Optional[float] = None
    # q1: Optional[float] = None
    # q3: Optional[float] = None
    
    numeric_stats: Optional[Dict[str, float]] = None
    
    n_outliers: Optional[int] = None  # Number of Outliers
    n_impossible_values: Optional[int] = None # Domain-violating Values
        
class DatasetProfile(BaseModel):
    n_rows: int # Shape of the Df
    n_columns: int # shape
    n_duplicate_rows: int # count of duplicated rows
    columns: List[ColumnProfile] # detailed per- column stats
    target_column: Optional[str] = None 
    target_class_distribution: Optional[Dict[str,int]] = None # imbalance detection
    notes: Optional[str] = None # comments by DQ Validation Agent
    
class DataQualityDecision(BaseModel):
    status: Literal["PASS", "WARN", "FAIL"] 
    summary: str 
    issues: List[str] = Field(default_factory=list)

# --------- PipelineState for LangGraph ---------

class PipelineState(TypedDict, total=False):
    df: Any  # pandas.DataFrame but we use Any to avoid import cycles
    # dataset_path:str
    
    # Data quality analysis
    dq_profile: DatasetProfile
    dq_decision: DataQualityDecision
    validation_report: str
    validation_status: str  # PASS/WARN/FAIL

    # Model
    model: Any
    metrics: Dict[str, float]
    shap_values: Any
    top_features: List[Dict[str, Any]]  
    # e.g., [{"name": "ejection_fraction", "importance": 0.34}]

    summary_markdown: str
    run_id: str
    

# ColumnProfile (Local to each feature)
    # ColumnProfile -> per Column stats and issues.
    # DatasetProfile -> Dataset-level Summary
    # dataQualityDecision -> The DQ agent's Verdict 
    # PipelineState -> The  mutable state that LangGraph nodes read/write

    # DQ agent builds one ColumnProfile per column
    # Whole collection become part of DatasetProfile 
    # The LLM gets this entire profile in JSON and produces the report.

    # Possible Extensions
    #     - median 
    #     - skewness
    #     - flagged_issues (per-column notes)

# DatasetProfile 
    # - cross-column properties
    # - class balance
    
    # Included n_duplicate_rows
        # Because duplicates are a classic data quality issue:
        # They can bias estimating prevalence of outcomes.
        # They can leak information across train/test splits.
        
# DataQualityDecision
    # This is the LLM’s final verdict about the dataset:
        # status: one of "PASS", "WARN", "FAIL".
        # Proceed with training
        # Proceed with caution
        # Stop pipeline (for dirty dataset)
        
    # summary: a markdown report describing:
        # Overview,
        # Column-level checks,
        # Row-level anomalies,
        # Recommendation.
        # issues: a concise list of bullet-point issues extracted by the LLM.