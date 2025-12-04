import os
import json
from pydantic import BaseModel
from agents.base import BaseAgent
from typing import Any, Dict, List, Optional
from core.state import PipelineState, DataQualityDecision


class ModelSummary(BaseModel):
    title: str
    executive_summary: str
    performance_section: str
    feature_importance_section: str
    data_quality_section: str   
    recommendations: str


class ModelOutputSummarizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="model_output_summarizer_agent")

    def run(self, state: PipelineState) -> PipelineState:
        # 1. Extract required info from state
        metrics: Dict[str, float] = state.get("metrics", {})  
        top_features: List[Dict[str, Any]] = state.get("top_features", []) 
        validation_status: str = state.get("validation_status", "UNKNOWN")
        dq_decision: Optional[DataQualityDecision] = state.get("dq_decision")  

        # Basic safety checks
        if not metrics or not top_features:
            # If we don't have real model outputs, just write a simple note.
            markdown = "# Model Summary\n\nModel metrics or feature importance not available."
            state["summary_markdown"] = markdown
            return state

        dq_summary = (
            dq_decision.summary if dq_decision is not None
            else "No data quality summary available."
        )

        # 2. Prepare a compact JSON payload for the LLM
        payload = {
            "validation_status": validation_status,
            "data_quality_summary": dq_summary,
            "metrics": metrics,
            "top_features": top_features[:10], 
            "task_description": "Binary classification: predicting DEATH_EVENT (0 = alive, 1 = death).",
        }

        # 3. Use structured output to get a ModelSummary object from the LLM
        structured_llm = self.llm.with_structured_output(ModelSummary)

        system_msg = {
            "role": "system",
            "content": (
                "You are a clinical machine learning assistant. "
                "Given model performance metrics, feature importance derived from SHAP values, "
                "and an upstream data quality summary, you generate clear, concise, and semi-technical "
                "summaries for clinicians and data scientists.\n\n"
                "You must produce a ModelSummary object with the following fields:\n"
                "- title: a short, descriptive title (e.g., 'Heart Failure Mortality Prediction Model Summary').\n"
                "- executive_summary: 2–4 sentences summarizing what the model predicts, how well it performs, "
                "  and a high-level statement about data quality.\n"
                "- performance_section: START with a short bullet list of metrics using the values from 'metrics' "
                "  (AUC, accuracy, F1), then add 2–3 sentences interpreting these metrics in plain language.\n"
                "  For example:\n"
                "    - **AUC:** 0.859\n"
                "    - **Accuracy:** 0.817\n"
                "    - **F1 score (positive class):** 0.667\n"
                "- feature_importance_section: explicitly state that feature importance is based on SHAP values, "
                "  then explain how the top features (in order) relate to risk (e.g., higher age increases risk, "
                "  lower ejection fraction increases risk, etc.). Make sure to mention some of the actual feature "
                "  names from 'top_features'.\n"
                "- data_quality_section: summarize the upstream data quality status (PASS/WARN/FAIL) taken from "
                "  'validation_status' and the most important issues from 'data_quality_summary', and explain how "
                "  they might affect trust in the model.\n"
                "- recommendations: suggest practical next steps and caveats for using this model in practice, "
                "  including any concerns about outliers, anomalies, imbalance, or a WARN/FAIL data quality status.\n\n"
                "Keep the tone professional, avoid heavy jargon, and assume the reader is a clinician or data scientist "
                "with some ML familiarity but not a deep ML expert."
            ),
        }

        user_msg = {
            "role": "user",
            "content": (
                "Here is the model information in JSON format:\n"
                f"{json.dumps(payload, indent=2)}\n\n"
                "Please analyze this and produce a ModelSummary."
            ),
        }

        summary: ModelSummary = structured_llm.invoke([system_msg, user_msg])

        # 4. Convert structured summary to markdown
        markdown = self._to_markdown(summary)

        state["summary_markdown"] = markdown

        # 5. write to a file in outputs/reports
        run_id = state.get("run_id", "unknown_run")
        os.makedirs("outputs/reports", exist_ok=True)
        output_path = os.path.join("outputs", "reports", f"model_summary_{run_id}.md")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        print(f"[ModelOutputSummarizerAgent] Saved model summary to: {output_path}")

        return state

    def _to_markdown(self, summary: ModelSummary) -> str:
        """
        Format the ModelSummary object into markdown text.
        """
        md: List[str] = []
        md.append(f"# {summary.title}\n")
        md.append("## Executive Summary\n")
        md.append(summary.executive_summary.strip() + "\n")
        md.append("## Performance\n")
        md.append(summary.performance_section.strip() + "\n")
        md.append("## Feature Importance\n")
        md.append(summary.feature_importance_section.strip() + "\n")
        md.append("## Data Quality Considerations\n")
        md.append(summary.data_quality_section.strip() + "\n")
        md.append("## Recommendations\n")
        md.append(summary.recommendations.strip() + "\n")
        return "\n".join(md)
