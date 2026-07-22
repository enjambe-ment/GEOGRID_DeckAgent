import anthropic
from pydantic import BaseModel


class AgentScore(BaseModel):
    structural_compliance: float  # 0-100: 고정 템플릿(포맷/큰틀) 준수도
    segment_customization: float  # 0-100: 고객 세그먼트 맞춤화 수준
    overall: float  # 0-100: 종합 점수
    feedback: str  # 다음 회차 개선 지시사항


class JudgeVerdict(BaseModel):
    scores: dict[str, AgentScore]  # key: agent name (claude/gpt/gemini)
    pairwise_structural_similarity: float  # 0-100: 세 초안 간 구조 일관성
    consensus_reached: bool
    best_agent: str
    summary: str


JUDGE_PROMPT_TEMPLATE = """\
다음은 하나의 제안서 템플릿(고정 포맷/큰틀)과 특정 고객 세그먼트에 대한 설명, \
그리고 서로 다른 AI 세 개가 생성한 제안서 초안입니다.

[고정 템플릿 / 포맷]
{template}

[고객 세그먼트]
{segment_brief}

[초안들]
{drafts_block}

각 초안(claude, gpt, gemini)에 대해 다음을 0~100점으로 채점하세요:
- structural_compliance: 위 고정 템플릿의 섹션 구성/순서/톤을 얼마나 정확히 따랐는가
- segment_customization: 해당 고객 세그먼트에 얼마나 잘 맞춤화되었는가
- overall: 위 두 점수를 종합한 최종 점수
- feedback: 다음 회차에서 개선해야 할 구체적인 지시사항 (한국어, 2~4문장). \
  이미 충분하면 "현재 수준 유지"라고 쓰세요.

또한 pairwise_structural_similarity에는 세 초안이 서로 얼마나 동일한 섹션 구성/순서를 \
따르고 있는지(구조적 일관성, 내용까지 같을 필요는 없음)를 0~100점으로 채점하세요.

consensus_reached는 아래 조건을 모두 만족할 때만 true로 설정하세요:
1. 세 초안의 overall 점수가 모두 {threshold} 이상
2. pairwise_structural_similarity가 {threshold} 이상

best_agent에는 overall 점수가 가장 높은 초안의 이름(claude/gpt/gemini 중 하나)을 넣으세요.
summary에는 이번 회차 평가에 대한 1~2문장 요약을 한국어로 넣으세요.
"""


class Judge:
    def __init__(self, model: str, threshold: float, api_key: str | None = None):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.threshold = threshold

    def evaluate(
        self, template: str, segment_brief: str, drafts: dict[str, str]
    ) -> JudgeVerdict:
        drafts_block = "\n\n".join(
            f"=== {name.upper()} 초안 ===\n{content}" for name, content in drafts.items()
        )
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            template=template,
            segment_brief=segment_brief,
            drafts_block=drafts_block,
            threshold=self.threshold,
        )
        response = self.client.messages.parse(
            model=self.model,
            max_tokens=4096,
            output_format=JudgeVerdict,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.parsed_output
