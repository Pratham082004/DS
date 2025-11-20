"""
RunOrchestratorAsync
--------------------
Main entry point to execute the AsyncOrchestrator pipeline end-to-end.
"""

import asyncio
import nest_asyncio
import traceback
import os
from orchestrator.orchestrator_async import AsyncOrchestrator

# Apply nested loop fix (important for Jupyter/Spyder)
nest_asyncio.apply()

# ------------------------ CONFIGURATION ------------------------ #
DATASET_PATH = "data/sample.csv"
TARGET = None        # <-- SET YOUR TARGET COLUMN HERE (e.g. "label", "price")
MAX_RETRIES = 2


# --------------------------------------------------------------- #
# ✅ Helper: extract model summary JSON path
# --------------------------------------------------------------- #
def get_model_summary_path(results):
    model_build = results.get("model_build")
    if not isinstance(model_build, dict):
        return None

    model_path = model_build.get("model_path")
    if not model_path:
        return None

    summary_path = model_path.replace(".joblib", "_summary.json")
    return summary_path if os.path.exists(summary_path) else None


# --------------------------------------------------------------- #
# ✅ Helper: extract insight report path
# --------------------------------------------------------------- #
def get_insight_report_path(results):
    insight = results.get("insight_report")
    if isinstance(insight, dict):
        return insight.get("insights_path")
    return None


# ------------------------ MAIN EXECUTION ------------------------ #
async def main():
    print("\n🚀 Starting AI Data Scientist Pipeline...\n")
    orchestrator = AsyncOrchestrator()

    try:
        results = await orchestrator.run_pipeline(DATASET_PATH, TARGET)

        # Extract correct paths
        model_summary_path = get_model_summary_path(results)
        insight_path = get_insight_report_path(results)

        print("\n=== ✅ Pipeline Execution Summary ===")
        print(f"📊 EDA Keys: {list(results.get('eda_summary', {}).keys()) or 'None'}")
        print(f"🧠 Model Summary JSON: {model_summary_path}")
        print(f"📝 Insight Report: {insight_path}")
        print("\n🎯 Full Pipeline Execution Completed Successfully!\n")

    except Exception as e:
        print("\n❌ Pipeline failed due to an unexpected error.")
        print("Error Details:", str(e))
        print(traceback.format_exc())


# ------------------------ RUNNER WRAPPER ------------------------ #
def safe_run():
    """Run asyncio safely with retry and nested loop support."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            loop = asyncio.get_event_loop()

            if loop.is_running():
                print(f"🔁 Event loop already running — scheduling pipeline (attempt {attempt})...")
                return asyncio.ensure_future(main())
            else:
                loop.run_until_complete(main())
                break

        except KeyboardInterrupt:
            print("\n🛑 Pipeline manually interrupted.")
            break

        except Exception as e:
            print(f"⚠️ Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print("⏳ Retrying...\n")
            else:
                print("❌ Max retries reached. Pipeline aborted.")
            continue


# ------------------------ ENTRY POINT ------------------------ #
if __name__ == "__main__":
    safe_run()
