from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver 

from .state import PipelineState
from agents.data_quality_validation_agent import DataQualityValidationAgent
from agents.model_output_summarizer_agent import ModelOutputSummarizerAgent 
from core.model_training import train_model_and_explain
from config.settings import DATASET_VARIANT, DATA_DIR 


dq_agent = DataQualityValidationAgent()
mos_agent = ModelOutputSummarizerAgent()  


def load_data_node(state: PipelineState) -> PipelineState:
    import os
    import pandas as pd

    if DATASET_VARIANT == "dirty":
        filename = "heart_failure_dirty.csv"
    else:
        filename = "heart_failure_clean.csv"

    path = os.path.join(DATA_DIR, filename)
    df = pd.read_csv(path)
    state["df"] = df
    print(f"[load_data_node] Loaded {DATASET_VARIANT} dataset '{filename}' with shape: {df.shape}")
    return state



def data_quality_validation_node(state: PipelineState) -> PipelineState:
    print("[data_quality_validation_node] Running Data Quality Validation Agent...")
    state = dq_agent.run(state)
    print(
        f"[data_quality_validation_node] Status = {state.get('validation_status')}"
    )
    return state


def train_and_infer_node(state: PipelineState) -> PipelineState:
    status = state.get("validation_status", "WARN")
    print(f"[train_and_infer_node] Starting with validation_status = {status}")

    if status == "FAIL":
        print("[train_and_infer_node] Validation status is FAIL. Skipping training.")
        state["metrics"] = {}
        state["top_features"] = []
        return state

    df = state.get("df")
    if df is None:
        raise ValueError("DataFrame missing from state in train_and_infer_node.")

    print("[train_and_infer_node] Training model and computing SHAP...")
    model, metrics, top_features, shap_values = train_model_and_explain(
        df, target_column="DEATH_EVENT"
    )

    state["model"] = model
    state["metrics"] = metrics
    state["top_features"] = top_features
    state["shap_values"] = shap_values

    print(f"[train_and_infer_node] Metrics: {metrics}")
    print(
        "[train_and_infer_node] Top features (by importance): "
        + ", ".join(f["name"] for f in top_features[:5])
    )

    return state


def model_output_summarizer_node(state: PipelineState) -> PipelineState:
    print("[model_output_summarizer_node] Running Model Output Summarizer Agent...")
    state = mos_agent.run(state)
    return state


def build_app():
    graph = StateGraph(PipelineState)

    graph.add_node("load_data", load_data_node)
    graph.add_node("data_quality_validation", data_quality_validation_node)
    graph.add_node("train_and_infer", train_and_infer_node)
    graph.add_node("model_output_summarizer", model_output_summarizer_node)

    graph.set_entry_point("load_data")

    graph.add_edge("load_data", "data_quality_validation")
    graph.add_edge("data_quality_validation", "train_and_infer")
    graph.add_edge("train_and_infer", "model_output_summarizer")
    graph.add_edge("model_output_summarizer", END)

    # Checkpointing can be added here later
    app = graph.compile()
    return app
