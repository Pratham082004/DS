"""
AsyncOrchestrator
-----------------
Coordinates DataAnalystAgent, ModelBuilderAgent, and InsightReporterAgent
in an asynchronous pipeline with optional DeepSeek/OpenRouter LLM integration.
"""

import asyncio
import os
from agents.data_analyst_agent import DataAnalystAgent
from agents.model_builder_agent import ModelBuilderAgent
from agents.insight_reporter_agent import InsightReporterAgent


class AsyncOrchestrator:
    def __init__(self, use_llm=False, llm_provider=None, llm_client=None):
        """
        use_llm: bool – whether to use a reasoning LLM (DeepSeek/OpenRouter)
        llm_provider: str – provider name ("deepseek", "openrouter", etc.)
        llm_client: callable – a function to send prompts and receive responses
        """
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_client = llm_client

        # Initialize all agents with LLM integration if available
        self.eda_agent = DataAnalystAgent(llm=llm_client)
        self.model_agent = ModelBuilderAgent(llm=llm_client)
        self.report_agent = InsightReporterAgent(llm=llm_client)

    # --------------------------- Async Wrappers --------------------------- #
    async def run_eda(self, dataset_path):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.eda_agent.process, dataset_path)

    async def run_model(self, dataset_path, target=None):
        loop = asyncio.get_event_loop()
        input_data = {"dataset_path": dataset_path}
        if target:
            input_data["target"] = target
        return await loop.run_in_executor(None, self.model_agent.process, input_data)

    async def run_report(self, eda_summary, model_summary):
        loop = asyncio.get_event_loop()

        model_path = model_summary.get("model_path")
        summary_json = model_path.replace(".pkl", "_summary.json") if model_path else None
        if not summary_json or not os.path.exists(summary_json):
            summary_json = None

        input_data = {
            "eda_path": "data/eda_summary.json",
            "model_path": summary_json
        }
        return await loop.run_in_executor(None, self.report_agent.process, input_data)

    # --------------------------- Full Pipeline --------------------------- #
    async def run_pipeline(self, dataset_path, target=None):
        print(f"[AsyncOrchestrator] 🚀 Starting pipeline (LLM={self.use_llm}, Provider={self.llm_provider})")

        print("[AsyncOrchestrator] → Running EDA Agent...")
        eda_future = asyncio.create_task(self.run_eda(dataset_path))
        eda_summary = await eda_future
        print("[AsyncOrchestrator] ✅ EDA completed.")

        print("[AsyncOrchestrator] → Running Model Builder Agent...")
        model_future = asyncio.create_task(self.run_model(dataset_path, target))
        model_summary = await model_future
        print("[AsyncOrchestrator] ✅ Model training completed.")

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
