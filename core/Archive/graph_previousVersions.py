# # core/graph.py

# from langgraph.graph import StateGraph, END
# from typing import cast

# from .state import PipelineState


# # --------- Node implementations (temporary stubs) ---------


# def load_data_node(state: PipelineState) -> PipelineState:
#     """
#     Load the tabular dataset into state["df"].

#     For now, we just load the 'clean' dataset.
#     Later we can parameterize this or allow switching to a dirty dataset.
#     """
#     import pandas as pd

#     df = pd.read_csv("data/Heart_failure_clinical_records_dataset.csv")
#     state["df"] = df
#     print(f"[load_data_node] Loaded dataset with shape: {df.shape}")
#     return state


# def data_quality_validation_node(state: PipelineState) -> PipelineState:
#     """
#     This will call our Data Quality Validation Agent.
#     For now, it's just a placeholder that we will replace.
#     """
#     print("[data_quality_validation_node] TODO: implement Data Quality Agent.")
#     # Temporary dummy values so the pipeline can proceed
#     state["validation_status"] = "WARN"
#     state["validation_report"] = "Data quality agent not implemented yet."
#     return state


# def train_and_infer_node(state: PipelineState) -> PipelineState:
#     """
#     This will train a simple model and compute metrics + SHAP.
#     We'll implement it after the Data Quality Agent.
#     """
#     print("[train_and_infer_node] TODO: implement model training + SHAP.")
#     state["metrics"] = {"auc": 0.0}
#     state["top_features"] = []
#     return state


# def model_output_summarizer_node(state: PipelineState) -> PipelineState:
#     """
#     This will call our Model Output Summarizer Agent (LLM).
#     """
#     print("[model_output_summarizer_node] TODO: implement model summarizer agent.")
#     state["summary_markdown"] = "# Model Summary\n\n(Placeholder.)"
#     return state


# # --------- Graph builder ---------


# def build_app():
#     """
#     Construct and compile the LangGraph application.
#     """
#     graph = StateGraph(PipelineState)

#     # Register nodes
#     graph.add_node("load_data", load_data_node)
#     graph.add_node("data_quality_validation", data_quality_validation_node)
#     graph.add_node("train_and_infer", train_and_infer_node)
#     graph.add_node("model_output_summarizer", model_output_summarizer_node)

#     # Entry point
#     graph.set_entry_point("load_data")

#     # Edges: define the order
#     graph.add_edge("load_data", "data_quality_validation")
#     graph.add_edge("data_quality_validation", "train_and_infer")
#     graph.add_edge("train_and_infer", "model_output_summarizer")
#     graph.add_edge("model_output_summarizer", END)

#     # TODO: later, we will add checkpointing/config here
#     app = graph.compile()

#     return app




# 1st update for data qulaity validation agent

# core/graph.py

# from langgraph.graph import StateGraph, END

# from .state import PipelineState
# from agents.data_quality_validation_agent import DataQualityValidationAgent

# # Instantiate the agent once (so it reuses the same LLM client)
# dq_agent = DataQualityValidationAgent()


# def load_data_node(state: PipelineState) -> PipelineState:
#     import pandas as pd

#     df = pd.read_csv("data/Heart_failure_clinical_records_dataset.csv")
#     state["df"] = df
#     print(f"[load_data_node] Loaded dataset with shape: {df.shape}")
#     return state


# def data_quality_validation_node(state: PipelineState) -> PipelineState:
#     """
#     Call the real Data Quality Validation Agent.
#     """
#     print("[data_quality_validation_node] Running Data Quality Validation Agent...")
#     state = dq_agent.run(state)
#     print(
#         f"[data_quality_validation_node] Status = {state.get('validation_status')}"
#     )
#     return state


# def train_and_infer_node(state: PipelineState) -> PipelineState:
#     print("[train_and_infer_node] TODO: implement model training + SHAP.")
#     state["metrics"] = {"auc": 0.0}
#     state["top_features"] = []
#     return state


# def model_output_summarizer_node(state: PipelineState) -> PipelineState:
#     print("[model_output_summarizer_node] TODO: implement model summarizer agent.")
#     state["summary_markdown"] = "# Model Summary\n\n(Placeholder.)"
#     return state


# def build_app():
#     graph = StateGraph(PipelineState)

#     graph.add_node("load_data", load_data_node)
#     graph.add_node("data_quality_validation", data_quality_validation_node)
#     graph.add_node("train_and_infer", train_and_infer_node)
#     graph.add_node("model_output_summarizer", model_output_summarizer_node)

#     graph.set_entry_point("load_data")

#     graph.add_edge("load_data", "data_quality_validation")
#     graph.add_edge("data_quality_validation", "train_and_infer")
#     graph.add_edge("train_and_infer", "model_output_summarizer")
#     graph.add_edge("model_output_summarizer", END)

#     app = graph.compile()
#     return app




# Update 2 Model Training
# core/graph.py

# from langgraph.graph import StateGraph, END

# from .state import PipelineState
# from agents.data_quality_validation_agent import DataQualityValidationAgent
# from core.model_training import train_model_and_explain  # <-- NEW IMPORT

# dq_agent = DataQualityValidationAgent()


# def load_data_node(state: PipelineState) -> PipelineState:
#     import pandas as pd

#     df = pd.read_csv("data/Heart_failure_clinical_records_dataset.csv")
#     state["df"] = df
#     print(f"[load_data_node] Loaded dataset with shape: {df.shape}")
#     return state


# def data_quality_validation_node(state: PipelineState) -> PipelineState:
#     print("[data_quality_validation_node] Running Data Quality Validation Agent...")
#     state = dq_agent.run(state)
#     print(
#         f"[data_quality_validation_node] Status = {state.get('validation_status')}"
#     )
#     return state


# def train_and_infer_node(state: PipelineState) -> PipelineState:
#     """
#     Train a model on the dataset (if validation isn't FAIL) and compute
#     metrics and feature importance (SHAP).
#     """
#     status = state.get("validation_status", "WARN")
#     print(f"[train_and_infer_node] Starting with validation_status = {status}")

#     # Simple rule: proceed even on WARN, but you could choose to skip on FAIL.
#     if status == "FAIL":
#         print("[train_and_infer_node] Validation status is FAIL. Skipping training.")
#         state["metrics"] = {}
#         state["top_features"] = []
#         return state

#     df = state.get("df")
#     if df is None:
#         raise ValueError("DataFrame missing from state in train_and_infer_node.")

#     print("[train_and_infer_node] Training model and computing SHAP...")
#     model, metrics, top_features, shap_values = train_model_and_explain(
#         df, target_column="DEATH_EVENT"
#     )

#     state["model"] = model
#     state["metrics"] = metrics
#     state["top_features"] = top_features
#     state["shap_values"] = shap_values

#     print(f"[train_and_infer_node] Metrics: {metrics}")
#     print(
#         "[train_and_infer_node] Top features (by importance): "
#         + ", ".join(f["name"] for f in top_features[:5])
#     )

#     return state


# def model_output_summarizer_node(state: PipelineState) -> PipelineState:
#     print("[model_output_summarizer_node] TODO: implement model summarizer agent.")
#     state["summary_markdown"] = "# Model Summary\n\n(Placeholder.)"
#     return state


# def build_app():
#     graph = StateGraph(PipelineState)

#     graph.add_node("load_data", load_data_node)
#     graph.add_node("data_quality_validation", data_quality_validation_node)
#     graph.add_node("train_and_infer", train_and_infer_node)
#     graph.add_node("model_output_summarizer", model_output_summarizer_node)

#     graph.set_entry_point("load_data")

#     graph.add_edge("load_data", "data_quality_validation")
#     graph.add_edge("data_quality_validation", "train_and_infer")
#     graph.add_edge("train_and_infer", "model_output_summarizer")
#     graph.add_edge("model_output_summarizer", END)

#     app = graph.compile()
#     return app

