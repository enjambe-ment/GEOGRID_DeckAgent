import argparse
import json
from pathlib import Path

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CONSENSUS_THRESHOLD,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
    GPT_MODEL,
    JUDGE_MODEL,
    MAX_ITERATIONS,
    OPENAI_API_KEY,
)
from pdf_utils import extract_text
from agents.claude_agent import ClaudeAgent
from agents.gemini_agent import GeminiAgent
from agents.gpt_agent import GPTAgent
from judge import Judge
from orchestrator import Orchestrator, extract_template


def main() -> None:
    parser = argparse.ArgumentParser(
        description="샘플 제안서 PDF를 분석해 포맷을 고정하고, 세그먼트별 맞춤 제안서를 "
        "Claude/GPT/Gemini 크로스체크 루프로 생성한다."
    )
    parser.add_argument("--pdf", required=True, help="샘플 제안서 PDF 경로")
    parser.add_argument("--segments", required=True, help="세그먼트 정의 JSON 경로")
    parser.add_argument("--output", default="output", help="결과 출력 디렉토리")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] 샘플 제안서 텍스트 추출 중...")
    sample_text = extract_text(args.pdf)

    claude = ClaudeAgent(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY)
    gpt = GPTAgent(model=GPT_MODEL, api_key=OPENAI_API_KEY)
    gemini = GeminiAgent(model=GEMINI_MODEL, api_key=GOOGLE_API_KEY)
    judge = Judge(model=JUDGE_MODEL, threshold=CONSENSUS_THRESHOLD, api_key=ANTHROPIC_API_KEY)

    print("[2/3] 고정 포맷/템플릿 추출 중 (Claude 기준)...")
    template = extract_template(sample_text, claude)
    (output_dir / "template.md").write_text(template, encoding="utf-8")

    segments = json.loads(Path(args.segments).read_text(encoding="utf-8"))

    orchestrator = Orchestrator(
        agents=[claude, gpt, gemini],
        judge=judge,
        max_iterations=MAX_ITERATIONS,
        output_dir=output_dir,
    )

    print("[3/3] 세그먼트별 제안서 생성 및 크로스체크 시작...")
    summary = []
    for seg in segments:
        print(f" - 세그먼트: {seg['name']}")
        result = orchestrator.run_segment(seg["name"], seg["brief"], template)
        summary.append(result)
        status = "합의 완료" if result["consensus_reached"] else "최대 반복 도달 (미합의)"
        print(f"   -> {status}, {result['iterations']}회 반복, 최종 채택: {result['final_agent']}")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n완료. 결과는 '{output_dir}' 디렉토리를 확인하세요.")


if __name__ == "__main__":
    main()
