import asyncio
from llm.llm_client import LLMClient


class DeepSeekR1:
    """
    Wrapper for DeepSeek-R1 model via OpenRouter.
    Provides higher-level reasoning and structured responses.
    """

    def __init__(self, model="deepseek/deepseek-r1:free"):
        self.client = LLMClient(model=model)

    async def reason(self, context, question, temperature=0.7):
        """
        Combines context + question to produce a structured reasoning output.
        Example use: generating insights or explaining EDA/model decisions.
        """
        messages = [
            {"role": "system", "content": "You are an expert AI data scientist with deep reasoning capabilities."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]

        response = await self.client.chat(messages, temperature=temperature, max_tokens=1500)
        return response.strip()

    async def summarize(self, text):
        """Summarize long outputs (like EDA or model reports)."""
        prompt = f"Summarize the following data analysis in concise technical terms:\n\n{text}"
        return await self.client.simple_prompt(prompt)

    async def generate_insights(self, eda_summary, model_summary):
        """
        Generates combined analytical insights from both EDA and model results.
        """
        context = (
            f"EDA Summary:\n{eda_summary}\n\n"
            f"Model Summary:\n{model_summary}\n\n"
            "Generate 3-5 clear, data-driven insights that summarize the dataset and model performance."
        )
        return await self.reason(context, "Summarize the insights in bullet points.")


# -------------------------- Local Test -------------------------- #
if __name__ == "__main__":
    async def test():
        ai = DeepSeekR1()
        context = "The dataset shows sales data across 3 regions with monthly trends."
        question = "What could be the possible reasons for declining sales in Q4?"
        answer = await ai.reason(context, question)
        print("\n🧠 DeepSeek-R1 Reasoning:\n", answer)

    asyncio.run(test())
