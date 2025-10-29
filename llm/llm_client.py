"""
LLMClient (Base Class)
----------------------
A simple reusable base for OpenRouter / DeepSeek API clients.
"""

import os
import requests

class LLMClient:
    def __init__(self, model: str, api_key: str = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("Missing OPENROUTER_API_KEY environment variable.")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/prathampatil1034/AI-Fullstack-Generator",
            "X-Title": "AI Data Science Orchestrator"
        }

    def query(self, prompt: str, system_message: str = "You are an expert data scientist."):
        """
        Send a prompt to the LLM and return its response.
        """
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1500
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[LLMClient] ❌ Error: {e}")
            return "LLM request failed or model unavailable."

    def __call__(self, prompt: str):
        return self.query(prompt)
