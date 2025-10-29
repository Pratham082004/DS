"""
Agent 3: InsightReporterAgent
-----------------------------
Combines EDA + Model summaries and generates reasoning-based insights using DeepSeek-R1.
Outputs:
- insights_report.txt
- insights_report.json
"""

import os
import json
import asyncio
from llm.deepseek_client import DeepSeekR1


class InsightReporterAgent:
    def __init__(self, llm=None):
        self.name = "InsightReporterAgent"
        self.output_dir = "data/reports"
        os.makedirs(self.output_dir, exist_ok=True)

        # Use provided LLM or fallback to DeepSeek
        self.llm = llm or DeepSeekR1()

    # --------------------------- Load Helpers --------------------------- #
    def _load_json(self, path):
        """Load JSON file if exists."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------- Main Process --------------------------- #
    def process(self, input_data):
        """
        input_data: dict
        {
            "eda_path": str,
            "model_path": str
        }
        """
        eda_path = input_data.get("eda_path")
        model_path = input_data.get("model_path")

        if not eda_path or not model_path:
            raise ValueError("Both 'eda_path' and 'model_path' are required.")

        print(f"[{self.name}] Loading summaries...")
        eda_summary = self._load_json(eda_path)
        model_summary = self._load_json(model_path)

        print(f"[{self.name}] Generating AI-driven insights...")
        insights = asyncio.run(self.llm.generate_insights(eda_summary, model_summary))

        # Save text and JSON versions
        report_txt_path = os.path.join(self.output_dir, "insights_report.txt")
        report_json_path = os.path.join(self.output_dir, "insights_report.json")

        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write(insights)

        report_data = {
            "eda_summary": eda_summary,
            "model_summary": model_summary,
            "ai_insights": insights
        }

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        print(f"[{self.name}] Report generated successfully.")
        return {
            "insights_path": report_txt_path,
            "report_json": report_json_path
        }


# --------------------------- Local Test --------------------------- #
if __name__ == "__main__":
    agent = InsightReporterAgent()

    test_input = {
        "eda_path": "data/eda_summary.json",
        "model_path": "data/models/sample_model_summary.json"
    }

    result = agent.process(test_input)
    print("\n✅ Insight Report Generated:\n", result)
