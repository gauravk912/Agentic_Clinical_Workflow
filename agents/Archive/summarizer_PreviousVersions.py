# # agents/model_output_summarizer_agent.py

# from typing import Any, Dict, List, Optional
# import json
# import os

# from pydantic import BaseModel
# from agents.base import BaseAgent
# from core.state import PipelineState, DataQualityDecision


# class ModelSummary(BaseModel):
#     """
#     Structured summary of model behavior, as returned by the LLM.
#     """
#     title: str
#     executive_summary: str
#     performance_section: str
#     feature_importance_section: str
#     recommendations: str


# class ModelOutputSummarizerAgent(BaseAgent):
#     """
#     Agent that turns raw model outputs (metrics + feature importance)
#     into a clear, human-readable markdown report.
#     """

#     def __init__(self):
#         super().__init__(name="model_output_summarizer_agent")

    
    
#     def run(self, state: PipelineState) -> PipelineState:
#         # 1. Extract required info from state
#         metrics: Dict[str, float] = state.get("metrics", {})  # type: ignore
#         top_features: List[Dict[str, Any]] = state.get("top_features", [])  # type: ignore
#         validation_status: str = state.get("validation_status", "UNKNOWN")
#         dq_decision: Optional[DataQualityDecision] = state.get("dq_decision")  # type: ignore

#         # Basic safety checks
#         if not metrics or not top_features:
#             # If we don't have real model outputs, just write a simple note.
#             markdown = "# Model Summary\n\nModel metrics or feature importance not available."
#             state["summary_markdown"] = markdown
#             return state

#         dq_summary = dq_decision.summary if dq_decision is not None else "No data quality summary available."

#         # 2. Prepare a compact JSON payload for the LLM
#         payload = {
#             "validation_status": validation_status,
#             "data_quality_summary": dq_summary,
#             "metrics": metrics,
#             "top_features": top_features[:10],  # limit to top 10 for brevity
#             "task_description": "Binary classification: predicting DEATH_EVENT (0 = alive, 1 = death).",
#         }

#         # 3. Use structured output to get a ModelSummary object from the LLM
#         structured_llm = self.llm.with_structured_output(ModelSummary)

#         system_msg = {
#             "role": "system",
#             "content": (
#                 "You are a clinical machine learning assistant. "
#                 "Given model performance metrics and feature importance, "
#                 "you generate clear, concise, and non-technical summaries for clinicians and stakeholders.\n\n"
#                 "You must produce a ModelSummary object with:\n"
#                 "- title: a short, descriptive title\n"
#                 "- executive_summary: 2–4 sentences summarizing the overall findings\n"
#                 "- performance_section: describe metrics (AUC, accuracy, F1) in plain language\n"
#                 "- feature_importance_section: explain top features and how they relate to risk\n"
#                 "- recommendations: suggest next steps or how to use this model responsibly."
#             ),
#         }

#         user_msg = {
#             "role": "user",
#             "content": (
#                 "Here is the model information in JSON format:\n"
#                 f"{json.dumps(payload, indent=2)}\n\n"
#                 "Please analyze this and produce a ModelSummary."
#             ),
#         }

#         summary: ModelSummary = structured_llm.invoke([system_msg, user_msg])

#         # 4. Convert structured summary to markdown
#         markdown = self._to_markdown(summary)

#         state["summary_markdown"] = markdown

#         # 5. Also write to a file in outputs/reports
#         run_id = state.get("run_id", "unknown_run")
#         os.makedirs("outputs/reports", exist_ok=True)
#         output_path = os.path.join("outputs", "reports", f"model_summary_{run_id}.md")
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(markdown)
#         print(f"[ModelOutputSummarizerAgent] Saved model summary to: {output_path}")

#         return state

#     def _to_markdown(self, summary: ModelSummary) -> str:
#         """
#         Format the ModelSummary object into markdown text.
#         """
#         md = []
#         md.append(f"# {summary.title}\n")
#         md.append("## Executive Summary\n")
#         md.append(summary.executive_summary.strip() + "\n")
#         md.append("## Performance\n")
#         md.append(summary.performance_section.strip() + "\n")
#         md.append("## Feature Importance\n")
#         md.append(summary.feature_importance_section.strip() + "\n")
#         md.append("## Recommendations\n")
#         md.append(summary.recommendations.strip() + "\n")
#         return "\n".join(md)

