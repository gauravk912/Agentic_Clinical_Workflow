# main.py

import os
from uuid import uuid4
from dotenv import load_dotenv

from core.graph import build_app
from core.state import PipelineState

load_dotenv()


def main():
    run_id = str(uuid4())
    print(f"[main] Starting run with run_id = {run_id}")

    app = build_app()

    # Initial state with run_id
    state = PipelineState(run_id=run_id)

    # --- NEW: config with thread_id for checkpointing ---
    config = {"configurable": {"thread_id": run_id}}

    # Invoke the graph with config so checkpoints are tied to this run
    final_state = app.invoke(state, config=config)

    # Pretty-print final outputs (same as you already do)
    print("\n=== FINAL STATE KEYS ===")
    print(final_state.keys())

    print("\n=== DATA QUALITY STATUS ===")
    print(final_state.get("validation_status"))

    print("\n=== DATA QUALITY REPORT (placeholder) ===")
    print(final_state.get("validation_report"))

    print("\n=== MODEL SUMMARY (placeholder) ===")
    print(final_state.get("summary_markdown"))


if __name__ == "__main__":
    main()
