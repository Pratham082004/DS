"""
AsyncOrchestrator
-----------------
Coordinates DataAnalystAgent, ModelBuilderAgent, and InsightReporterAgent
in an asynchronous pipeline without any LLM dependencies.
"""

import asyncio
import os
from agents.data_analyst_agent import DataAnalystAgent
from agents.model_builder_agent import ModelBuilderAgent
from agents.insight_reporter_agent import InsightReporterAgent


class AsyncOrchestrator:
    def __init__(self):
        """Initialize all agents without LLM integration."""
        self.eda_agent = DataAnalystAgent()
        self.model_agent = ModelBuilderAgent()
        self.report_agent = InsightReporterAgent()

    # --------------------------- Async Wrappers --------------------------- #
    async def run_eda(self, dataset_path):
        """Run the EDA agent asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.eda_agent.process, dataset_path)

    async def run_model(self, dataset_path, target=None):
        """Run the Model Builder agent asynchronously."""
        loop = asyncio.get_event_loop()
        input_data = {"dataset_path": dataset_path}
        if target:
            input_data["target"] = target
        return await loop.run_in_executor(None, self.model_agent.process, input_data)

    async def run_report(self, eda_summary, model_summary):
        """Run the Insight Reporter agent asynchronously."""
        loop = asyncio.get_event_loop()

        model_path = model_summary.get("model_path")
        summary_json = model_path.replace(".pkl", "_summary.json") if model_path else None

        print(f"[AsyncOrchestrator] Checking model summary path: {summary_json}")

        if not summary_json or not os.path.exists(summary_json):
            print("⚠️ Model summary JSON not found. Skipping insight generation.")
            return {"insights_path": None, "report_json": None}

        if not os.path.exists("data/eda_summary.json"):
            print("⚠️ EDA summary JSON not found. Skipping insight generation.")
            return {"insights_path": None, "report_json": None}

        input_data = {
            "eda_path": "data/eda_summary.json",
            "model_path": summary_json
        }

        print("[AsyncOrchestrator] Running InsightReporterAgent...")
        return await loop.run_in_executor(None, self.report_agent.process, input_data)

    # --------------------------- Full Pipeline --------------------------- #
    async def run_pipeline(self, dataset_path, target=None):
        """Run the full data-to-model-to-insight pipeline."""
        print("[AsyncOrchestrator] 🚀 Starting pipeline")

        # --- Step 1: Run EDA Agent ---
        print("[AsyncOrchestrator] → Running EDA Agent...")
        eda_future = asyncio.create_task(self.run_eda(dataset_path))
        eda_summary = await eda_future
        print("[AsyncOrchestrator] ✅ EDA completed.")

        # Save EDA summary to file
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/eda_summary.json", "w", encoding="utf-8") as f:
                import json
                json.dump(eda_summary, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save EDA summary: {e}")

        # --- Step 2: Run Model Builder Agent ---
        print("[AsyncOrchestrator] → Running Model Builder Agent...")
        model_future = asyncio.create_task(self.run_model(dataset_path, target))
        model_summary = await model_future
        print("[AsyncOrchestrator] ✅ Model training completed.")

        # --- Step 3: Run Insight Reporter Agent ---
        print("[AsyncOrchestrator] → Generating Insights Report...")
        report_future = asyncio.create_task(self.run_report(eda_summary, model_summary))
        report_result = await report_future
        print("[AsyncOrchestrator] ✅ Insights report generated.")

        print("[AsyncOrchestrator] 🎯 Pipeline completed successfully!")

        return {
            "eda_summary": eda_summary,
            "model_summary": model_summary,
            "report_result": report_result
        }
