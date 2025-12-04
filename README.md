# Agentic Clinical ML Workflow – Data Quality Validation & Model Output Summarization

This project implements a **mini agentic workflow** for a tabular clinical ML task using **LangGraph + LangChain + ML + LLMs**.

It is designed as a standalone case-study implementation for:

> **Two new agents** that extend an agentic ML workflow:
>
> 1. **Data Quality Validation Agent** – Checks input datasets for missing values, anomalies, and type consistency before processing. Produces a validation report and flags critical issues.
> 2. **Model Output Summarizer Agent** – Converts complex prediction outputs and SHAP style explanations into clear, human-readable summaries.

The pipeline operates on a **heart failure clinical records dataset** (tabular) and demonstrates:

- Agentic orchestration with **LangGraph**
- **Pre-model** data quality checks (rules + anomaly detection + LLM reasoning)
- **Post-model** summarization of metrics and SHAP-based feature importance via an LLM with structured output
- Clean integration and conditional flow based on agent decisions

---

## 1. High-Level Overview

### ML Task

- **Task:** Binary classification
- **Target:** `DEATH_EVENT`
  - `0` = patient survived
  - `1` = patient died
- **Data:** Heart failure clinical records dataset (tabular, numeric features)

### Agents Implemented

#### 1. Data Quality Validation Agent

**Goal:** Decide if a dataset is safe to model on and explain *why*.

It:

- Analyzes the dataset column-wise and dataset-wide
- Uses rule-based checks:
  - Missing values (per column)
  - Type consistency (numeric / categorical / mixed)
  - Outliers via IQR
  - Impossible values (e.g., negative lab values where not allowed)
  - Duplicate rows
- Trains a **row-level anomaly detector** using `IsolationForest`
- Delegates the final decision to an **LLM** using **structured output** (`DataQualityDecision`):
  - `status`: `PASS | WARN | FAIL`
  - `summary`: markdown data quality report
  - `issues`: list of key issues

It writes:

- Structured objects into shared state: `dq_profile`, `dq_decision`, `validation_status`
- A markdown report to disk:
  `outputs/reports/data_quality_report_<run_id>.md`

---

#### 2. Model Output Summarizer Agent

**Goal:** Explain what the trained model is doing in human language.

It:

- Reads from state:
  - `metrics` (AUC, accuracy, F1, etc.)
  - `top_features` (SHAP-based global importance, `{name, importance}`)
  - `validation_status` (from DQ agent)
  - `dq_decision.summary` (full DQ markdown report)
- Packages this into a compact JSON payload
- Calls an **LLM with structured output** to get a `ModelSummary` object with fields:
  - `title`
  - `executive_summary`
  - `performance_section`
  - `feature_importance_section`
  - `data_quality_section`
  - `recommendations`
- Renders the structured summary into markdown and saves it to:
  - `state["summary_markdown"]`
  - `outputs/reports/model_summary_<run_id>.md`

This directly addresses:

> *“Converts complex prediction outputs and SHAP explanations into clear, human-readable summaries.”*

---

## 2. Architecture & Workflow

The workflow is implemented using **LangGraph** and a typed shared state (`PipelineState`).

### PipelineState

Defined in `core/state.py` as a `TypedDict` with fields such as:

- `df`: the loaded `pandas.DataFrame`
- `dq_profile`: `DatasetProfile` (Pydantic model for dataset stats)
- `dq_decision`: `DataQualityDecision` (`status`, `summary`, `issues`)
- `validation_report`: markdown string
- `validation_status`: `"PASS" | "WARN" | "FAIL"`
- `model`: trained sklearn Pipeline
- `metrics`: dict of evaluation metrics
- `shap_values`: SHAP explanation object
- `top_features`: list of `{"name": ..., "importance": ...}`
- `summary_markdown`: final human-readable model summary
- `run_id`: unique identifier per run

### LangGraph Workflow (`core/graph.py`)

The graph is **linear** but agentic:

```text
load_data
   ↓
data_quality_validation
   ↓
train_and_infer
   ↓
model_output_summarizer
   ↓
END
```


Nodes:

1. **`load_data_node`**
   * Chooses dataset variant based on environment:
     * `DATASET_VARIANT=clean` → `data/heart_failure_clean.csv`
     * `DATASET_VARIANT=dirty` → `data/heart_failure_dirty.csv`
   * Loads into `state["df"]`.
2. **`data_quality_validation_node`**
   * Calls `DataQualityValidationAgent.run(state)`
   * Produces:
     * `dq_profile`
     * `dq_decision`
     * `validation_status` (`PASS/WARN/FAIL`)
     * Data quality markdown report (saved to disk)
3. **`train_and_infer_node`**
   * Reads `validation_status`:
     * If `FAIL`: **skips training** and sets `metrics = {}`, `top_features = []`.
     * Else:
       * Trains **Logistic Regression** on the dataset.
       * Evaluates on a held-out test set.
       * Computes SHAP-based global feature importance.
   * Writes:
     * `model`, `metrics`, `top_features`, `shap_values`.
4. **`model_output_summarizer_node`**
   * Calls `ModelOutputSummarizerAgent.run(state)`
   * Produces:
     * `summary_markdown`
     * Saves `model_summary_<run_id>.md` to disk.

The graph is compiled in `build_app()` and invoked from `main.py`.


## 3. Key Components

### 3.1 Agents

#### BaseAgent (`agents/base.py`)

* Provides a common LLM client using `ChatOpenAI`.
* Reads:
  * `OPENAI_API_KEY`
  * `LLM_MODEL_NAME`
* Exposes a helper `call_llm(messages)` and structured-output access via `.with_structured_output(...)`.

#### DataQualityValidationAgent (`agents/data_quality_validation_agent.py`)

* Builds a **DatasetProfile** using Pydantic models:
  * `ColumnProfile`: schema for per-column stats & numeric summary.
  * `DatasetProfile`: high-level dataset metadata.
* Performs:
  * Missingness analysis per column.
  * Type inference (`numeric`, `categorical`, `mixed`).
  * Outlier detection via IQR (for numeric columns).
  * Domain-specific impossible value checks (e.g., non-negative clinical columns).
  * Duplicate row detection.
* Trains an **IsolationForest** on numeric features for row-level anomaly detection (`core/anomaly_detection.py`).
* Aggregates issues into a list and then calls the LLM with structured output:
  * Target model: `DataQualityDecision` (`status`, `summary`, `issues`).
  * `summary` is a  **well-structured markdown report** :
    * `# Data Quality Report`
    * `## Overview`
    * `## Column-level checks`
    * `## Row-level anomalies`
    * `## Recommendation`
    * `## Issues`

#### ModelOutputSummarizerAgent (`agents/model_output_summarizer_agent.py`)

* Consumes:
  * `metrics`
  * `top_features` (from SHAP)
  * `validation_status`
  * `dq_decision.summary`
* Defines a Pydantic model `ModelSummary`:
  * `title`
  * `executive_summary`
  * `performance_section`
  * `feature_importance_section`
  * `data_quality_section`
  * `recommendations`
* Uses `self.llm.with_structured_output(ModelSummary)` to get a **schema-validated response** from the LLM.
* Converts the `ModelSummary` to markdown with fixed sections:
  * `# <title>`
  * `## Executive Summary`
  * `## Performance`
  * `## Feature Importance`
  * `## Data Quality Considerations`
  * `## Recommendations`
* Saves the report to `outputs/reports/model_summary_<run_id>.md`.

---

### 3.2 Model Training & Explainability (`core/model_training.py`)

* Uses scikit-learn to train:
  * Pipeline: `StandardScaler` + `LogisticRegression(max_iter=1000)`
* Train/test split:
  * `train_test_split` with `stratify=y` for label balance.
* Metrics:
  * `roc_auc_score`
  * `accuracy_score`
  * `f1_score`
* Explainability (SHAP):
  * Uses `shap.LinearExplainer` with the trained Logistic Regression.
  * Computes SHAP values for test set.
  * Aggregates mean absolute SHAP per feature to build global importance ranking.
  * Outputs:
    * `top_features`: sorted list of `{"name": feature_name, "importance": mean_abs_shap}`.
    * `shap_values`: SHAP `Explanation` object (available for further analysis if needed).

---

### 3.3 Anomaly Detection (`core/anomaly_detection.py`)

* Uses `IsolationForest` (unsupervised) for row-level anomaly detection.
* Excludes the target column (`DEATH_EVENT`) from features.
* Returns:
  * Trained model (optional).
  * Summary dict:
    * `n_samples`, `n_features`
    * `n_anomalies`, `anomaly_fraction`
    * `anomaly_score_min/max/mean`

The Data Quality Agent uses `anomaly_fraction` to decide if anomaly rate is concerning and feeds that into the LLM.

---

## 4. Dataset & Variants

* Base dataset: **Heart Failure Clinical Records Dataset** (tabular, numeric features).
* Located in `data/`:
  * `heart_failure_clean.csv`
    * Cleaned version: no missing values, no impossible negatives.
  * `heart_failure_dirty.csv`
    * Synthetic “dirty” version:
      * Missing values in key columns.
      * Negative values in non-negative clinical features.
      * Duplicate rows.
      * Missing values in the target column.

The `DATASET_VARIANT` environment variable controls which dataset is used at runtime.

---

## 5. Setup & Installation

### 5.1 Prerequisites

* **Python 3.10+** recommended
* A valid **OpenAI API key**

### 5.2 Create and activate a virtual environment

<pre class="overflow-visible!" data-start="9951" data-end="10069"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span># From project root</span><span>
python3 -m venv .venv
</span><span>source</span><span> .venv/bin/activate   </span><span># On Windows: .venv\Scripts\activate</span><span>
</span></span></code></div></div></pre>

### 5.3 Install dependencies

<pre class="overflow-visible!" data-start="10101" data-end="10144"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>pip install -r requirements.txt
</span></span></code></div></div></pre>

### 5.4 Environment variables

Create a `.env` file in the project root:

<pre class="overflow-visible!" data-start="10220" data-end="10489"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-env"><span>OPENAI_API_KEY=sk-...your_key_here...
LLM_MODEL_NAME=gpt-4.1-mini

# Data options
DATASET_VARIANT=clean    # or "dirty"
DATA_DIR=data

# Data quality thresholds
DQ_MISSINGNESS_WARN_THRESHOLD=0.1
DQ_MISSINGNESS_FAIL_THRESHOLD=0.3
DQ_OUTLIER_WARN_THRESHOLD=0.1
</span></code></div></div></pre>

> You can also create a `.env.example` to share with reviewers and keep your real key only in `.env`.

---

## 6. How to Run

### 6.1 Clean dataset run

<pre class="overflow-visible!" data-start="10644" data-end="10726"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>source</span><span> .venv/bin/activate
</span><span>export</span><span> DATASET_VARIANT=clean
python3 main.py
</span></span></code></div></div></pre>

You should see logs like:

<pre class="overflow-visible!" data-start="10755" data-end="11511"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>[main] Starting run with run_id = ...
[load_data_node] Loaded clean dataset 'heart_failure_clean.csv' with shape: (299, 13)
[data_quality_validation_node] Running Data Quality Validation Agent...
[DataQualityValidationAgent] Saved DQ report to: outputs/reports/data_quality_report_<run_id>.md
[DataQualityValidationAgent] Status = WARN
[train_and_infer_node] Starting with validation_status = WARN
[train_and_infer_node] Training model and computing SHAP...
[train_and_infer_node] Metrics: {...}
[train_and_infer_node] Top features (by importance): time, ejection_fraction, ...
[model_output_summarizer_node] Running Model Output Summarizer Agent...
[ModelOutputSummarizerAgent] Saved model summary to: outputs/reports/model_summary_<run_id>.md
</span></span></code></div></div></pre>

Outputs:

* `outputs/reports/data_quality_report_<run_id>.md`
* `outputs/reports/model_summary_<run_id>.md`

### 6.2 Dirty dataset run

<pre class="overflow-visible!" data-start="11649" data-end="11731"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>source</span><span> .venv/bin/activate
</span><span>export</span><span> DATASET_VARIANT=dirty
python3 main.py
</span></span></code></div></div></pre>

You should see:

* Data Quality Agent detects:
  * High missingness in key columns,
  * Impossible negative values,
  * Duplicates,
  * High anomaly fraction.
* `validation_status` becomes `FAIL`.
* `train_and_infer_node` skips training.
* Model Summary Agent produces a placeholder or a more cautious narrative (depending on configuration).

---

## 7. Folder Structure

A typical project layout:

<pre class="overflow-visible!" data-start="12132" data-end="12774"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-text"><span><span>agentic_case_study/
├── main.py
├── requirements.txt
├── README.md
├── .env                # not committed
├── config/
│   └── settings.py
├── core/
│   ├── state.py
│   ├── graph.py
│   ├── model_training.py
│   └── anomaly_detection.py
├── agents/
│   ├── base.py
│   ├── data_quality_validation_agent.py
│   ├── model_output_summarizer_agent.py
│   └── tools.py
├── data/
│   ├── heart_failure_clean.csv
│   └── heart_failure_dirty.csv
└── outputs/
    ├── reports/
    │   ├── data_quality_report_<run_id>.md
    │   └── model_summary_<run_id>.md
    └── debug/
        └── dq_profile_<run_id>.json   # optional debug profiles
</span></span></code></div></div></pre>

---

## 8. How This Extends the Agentic Workflow (Case Study Fit)

This project extends an agentic ML pipeline in two important ways:

1. **Pre-model Data Quality Gate (Data Quality Validation Agent)**
   * Encapsulates data-quality reasoning into a dedicated agent.
   * Produces both a **structured decision** (`PASS/WARN/FAIL`) and a  **human-readable report** .
   * Controls whether downstream modeling is allowed (no training on FAIL).
2. **Post-model Explanatory Layer (Model Output Summarizer Agent)**
   * Consumes metrics, SHAP/LIME-style feature importance, and DQ context.
   * Uses **LLM structured output** to create consistent, markdown-based model reports.
   * Bridges model behavior and data quality, directly targeting clinician / stakeholder interpretability.

Together, these agents address:

* **Quality control** before modeling.
* **Interpretability and communication** after modeling.
* **Coordination across the pipeline** (validation status informs training; data quality informs summary).

---

## 9. Extensibility & Future Work

Possible extensions (useful for critique / future roadmap):

* **Audit Trail Agent** :
  * Log every node’s input, output, and decision to a database (e.g., SQLite or Postgres).
  * Provide a queryable history of agent actions.
* **Error Recovery Agent** :
  * Detect failed agent calls or missing outputs.
  * Automatically retry step(s) or route to a fallback path.
* **Explainability Agent (Per-patient)** :
  * Generate SHAP plots or textual explanations for individual patients.
  * Offer “patient-level” narratives alongside global feature importance.
* **Multimodal Extension** :
  * Extend from purely tabular data to scenarios where imaging or text is also present.
  * Add agents specialized for imaging/text preprocessing and explanation.
