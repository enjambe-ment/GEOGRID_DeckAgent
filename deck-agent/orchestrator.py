import concurrent.futures as cf
import json
from pathlib import Path

from agents.base import Agent
from judge import Judge


TEMPLATE_EXTRACTION_PROMPT = """\
아래는 제안서 샘플 문서의 전체 텍스트입니다. 이 문서를 분석하여, 앞으로 모든 고객 \
세그먼트별 제안서 제작 시 반드시 고정해야 할 "포맷과 큰 틀"을 정리하세요.

포함할 내용:
1. 섹션 목록과 순서
2. 각 섹션의 목적과 필수 포함 요소
3. 전체적인 톤 앤 매너, 문체 가이드
4. 분량/구성 비율 가이드

세그먼트별로 달라져도 되는 부분(고객사명, 산업별 사례, 가격 옵션 등)은 별도로 명시하세요.
결과는 다른 AI가 그대로 지시사항으로 사용할 수 있도록 마크다운으로 명확하게 작성하세요.

[샘플 제안서 원문]
{sample_text}
"""

DRAFT_PROMPT_TEMPLATE = """\
당신은 제안서 작성 전문가입니다. 아래 고정 템플릿(포맷/큰틀)을 반드시 그대로 따르면서, \
주어진 고객 세그먼트에 맞게 내용을 맞춤화한 제안서를 작성하세요.

[고정 템플릿 / 포맷 — 반드시 준수]
{template}

[고객 세그먼트 정보]
{segment_brief}

[이전 회차 피드백 — 있다면 반드시 반영]
{revision_note}

전체 제안서를 마크다운으로 작성하세요. 템플릿의 섹션 구성과 순서를 임의로 바꾸지 마세요.
"""

DRAFT_SYSTEM_PROMPT = "당신은 B2B 제안서 작성 전문가입니다. 지시받은 포맷을 절대 임의로 바꾸지 않습니다."


def extract_template(sample_text: str, template_agent: Agent) -> str:
    prompt = TEMPLATE_EXTRACTION_PROMPT.format(sample_text=sample_text)
    return template_agent.generate(
        system="당신은 제안서 포맷/구조를 분석하는 전문가입니다.", prompt=prompt
    )


class Orchestrator:
    def __init__(
        self,
        agents: list[Agent],
        judge: Judge,
        max_iterations: int,
        output_dir: Path,
    ):
        self.agents = agents
        self.judge = judge
        self.max_iterations = max_iterations
        self.output_dir = output_dir

    def _generate_all(
        self, template: str, segment_brief: str, revision_notes: dict[str, str]
    ) -> dict[str, str]:
        def call(agent: Agent) -> tuple[str, str]:
            note = revision_notes.get(agent.name, "없음 (첫 회차)")
            prompt = DRAFT_PROMPT_TEMPLATE.format(
                template=template, segment_brief=segment_brief, revision_note=note
            )
            return agent.name, agent.generate(system=DRAFT_SYSTEM_PROMPT, prompt=prompt)

        with cf.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            results = dict(executor.map(call, self.agents))
        return results

    def run_segment(self, segment_name: str, segment_brief: str, template: str) -> dict:
        revision_notes: dict[str, str] = {}
        history: list[dict] = []
        verdict = None
        drafts: dict[str, str] = {}

        for iteration in range(1, self.max_iterations + 1):
            drafts = self._generate_all(template, segment_brief, revision_notes)
            verdict = self.judge.evaluate(template, segment_brief, drafts)

            history.append(
                {
                    "iteration": iteration,
                    "scores": {k: v.model_dump() for k, v in verdict.scores.items()},
                    "pairwise_structural_similarity": verdict.pairwise_structural_similarity,
                    "consensus_reached": verdict.consensus_reached,
                    "summary": verdict.summary,
                }
            )

            if verdict.consensus_reached:
                break

            revision_notes = {name: score.feedback for name, score in verdict.scores.items()}

        segment_dir = self.output_dir / segment_name
        segment_dir.mkdir(parents=True, exist_ok=True)

        for name, content in drafts.items():
            (segment_dir / f"{name}.md").write_text(content, encoding="utf-8")

        final_name = verdict.best_agent if verdict and verdict.best_agent in drafts else next(iter(drafts))
        (segment_dir / "FINAL.md").write_text(drafts[final_name], encoding="utf-8")
        (segment_dir / "iteration_log.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        return {
            "segment": segment_name,
            "consensus_reached": bool(verdict and verdict.consensus_reached),
            "iterations": len(history),
            "final_agent": final_name,
        }
