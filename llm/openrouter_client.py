import os
import aiohttp
import asyncio
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    A generic async LLM client for OpenRouter-compatible APIs.
    Works with DeepSeek-R1 or any other OpenRouter-supported model.
    """

    def __init__(self, model="deepseek/deepseek-r1:free"):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

        if not self.api_key:
            raise ValueError("❌ OPENROUTER_API_KEY not found in environment. Please add it to your .env file.")

    async def chat(self, messages, temperature=0.7, max_tokens=1024):
        """
        Send messages to the OpenRouter LLM asynchronously.
        messages: list of {"role": "user"/"system"/"assistant", "content": "..."}
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost/",
            "X-Title": "DataScienceAgent",
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(self.base_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"OpenRouter API error [{response.status}]: {text}")

                data = await response.json()
                return data["choices"][0]["message"]["content"]

    async def simple_prompt(self, prompt):
        """Shortcut for single-turn interactions."""
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages)

# -------------------------- Test Locally -------------------------- #
if __name__ == "__main__":
    async def test():
        client = LLMClient()
        reply = await client.simple_prompt("List 3 insights from the Iris dataset in short.")
        print("\nLLM Reply:\n", reply)

    asyncio.run(test())
