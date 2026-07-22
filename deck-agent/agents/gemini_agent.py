import google.generativeai as genai

from .base import Agent


class GeminiAgent(Agent):
    name = "gemini"

    def __init__(self, model: str, api_key: str | None = None):
        genai.configure(api_key=api_key)
        self._model_name = model

    def generate(self, system: str, prompt: str) -> str:
        # system_instruction이 바뀔 수 있으므로 호출마다 모델 핸들을 새로 만든다.
        model = genai.GenerativeModel(self._model_name, system_instruction=system)
        response = model.generate_content(prompt)
        return response.text or ""