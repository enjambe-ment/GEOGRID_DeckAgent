from openai import OpenAI

from .base import Agent


class GPTAgent(Agent):
    name = "gpt"

    def __init__(self, model: str, api_key: str | None = None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""
