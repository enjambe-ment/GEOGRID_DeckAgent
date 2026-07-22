import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# NOTE: GPT_MODEL / GEMINI_MODEL 기본값은 작성 시점 기준이므로,
# 실제 사용 전 OpenAI / Google 콘솔에서 최신 모델 ID를 확인하세요.
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-8")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-pro")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-opus-4-8")

MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "5"))
CONSENSUS_THRESHOLD = float(os.environ.get("CONSENSUS_THRESHOLD", "85"))
