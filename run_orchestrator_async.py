import asyncio
import nest_asyncio
from orchestrator.orchestrator_async import AsyncOrchestrator
from llm.openrouter_client import LLMClient

# Apply nested event loop fix for Spyder/Jupyter
nest_asyncio.apply()

dataset_path = "data/sample.csv"
target = None

# Initialize the LLM client
llm_client = LLMClient(model="deepseek-r1:latest")

# Pass the LLM into the orchestrator
orc = AsyncOrchestrator(
    use_llm=True,
    llm_provider="deepseek",
    llm_client=llm_client
)

async def main():
    results = await orc.run_pipeline(dataset_path, target)
    print("\n=== Pipeline Completed ===")
    print("EDA Keys:", results["eda_summary"].keys())
    print("Model Summary:", results["model_summary"])
    print("Report Files:", results["report_result"])

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("🔁 Event loop already running — executing using create_task()...")
            task = loop.create_task(main())
        else:
            asyncio.run(main())
    except Exception as e:
        print(f"❌ Error while running orchestrator: {e}")
