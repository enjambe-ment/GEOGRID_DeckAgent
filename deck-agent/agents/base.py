from abc import ABC, abstractmethod


class Agent(ABC):
    """서로 다른 AI 제공자를 동일한 인터페이스로 감싸는 베이스 클래스."""

    name: str

    @abstractmethod
    def generate(self, system: str, prompt: str) -> str:
        """system 지시문과 user prompt를 받아 텍스트 응답을 반환한다."""
        raise NotImplementedError
