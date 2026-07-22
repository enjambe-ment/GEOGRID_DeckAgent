import anthropic

from .base import Agent


class ClaudeAgent(Agent):
    name = "claude"

    def __init__(self, model: str, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")
