"""LLM-as-judge for FAITHFULNESS: is every claim in the answer supported by the excerpts it was given?

Runs only in evals (offline), never in the user's request path. A stronger model than the one that
answered is used, to reduce "grading its own homework" bias. The judge returns a verdict PER CLAIM
with a reason; our code computes the score from those verdicts (we don't ask the model for a number).
"""
from google import genai
from google.genai import types
from pydantic import BaseModel

from hr_assistant.core.retry import google_api_retry
from hr_assistant.retrieval.search import RetrievedChunk

JUDGE_SYSTEM_PROMPT = """\
You are a strict reviewer checking an HR assistant's answer against the policy excerpts it was given.

Split the answer into individual factual claims. For each claim decide if the excerpts SUPPORT it:
- Supported: the excerpts state it, or it follows directly and unambiguously from them.
- NOT supported: it is missing from the excerpts, contradicts them, or ADDS anything to them:
  extra emphasis or scope ("under any circumstances", "always", "never", "all employees"),
  extra conditions, exceptions, numbers, examples, or advice that the excerpts do not contain.
- NOT supported (stricter checks, added after human review of the judge):
  - a paraphrase that is LESS precise or ambiguous than the excerpt, e.g. "bi-weekly" for "every two weeks"
    ("bi-weekly" can also mean twice a week);
  - a conclusion about the asker's personal situation that depends on facts nobody stated, e.g. "your
    three-week request is covered" when the policy gives 30 days per year and the asker's used days are unknown.
- Supported: simple arithmetic and basic, uncontroversial world knowledge that is only used to APPLY the
  excerpts to the question (e.g. "Kerala is in India", "three weeks is 21 days"). This is not new policy.
- A claim that something is NOT mentioned in the excerpts is supported if the excerpts indeed don't mention it.
- Ignore citation markers like [1][2], greetings, and politeness.

If expected facts are given, also say whether the answer states them (meaning, not exact words).
Judge only against the excerpts, never against your own knowledge.
"""


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool
    excerpt: int | None  # which excerpt supports it (None if unsupported)
    reason: str


class JudgeVerdict(BaseModel):
    claims: list[ClaimVerdict]
    expected_facts_stated: bool | None  # None when no expected facts were given

    @property
    def unsupported(self) -> list[ClaimVerdict]:
        return [c for c in self.claims if not c.supported]

    @property
    def faithful(self) -> bool:
        return not self.unsupported  # computed by us from per-claim verdicts


def build_judge_prompt(question: str, chunks: list[RetrievedChunk], answer: str, expected_facts: list[str]) -> str:
    excerpts = "\n\n".join(
        f'<excerpt id="{n}" source="{c.heading_path}">\n{c.content}\n</excerpt>' for n, c in enumerate(chunks, 1)
    )
    facts = "\n".join(f"- {f.replace('|', ' OR ')}" for f in expected_facts) or "(none)"
    return (
        f"<excerpts>\n{excerpts}\n</excerpts>\n\n<question>\n{question}\n</question>\n\n"
        f"<answer>\n{answer}\n</answer>\n\n<expected_facts>\n{facts}\n</expected_facts>"
    )


class Judge:
    def __init__(self, *, project: str, location: str = "global", model: str = "gemini-3.1-pro-preview") -> None:
        self.model = model
        self._client = genai.Client(vertexai=True, project=project, location=location)

    @google_api_retry
    async def judge(
        self, question: str, chunks: list[RetrievedChunk], answer: str, expected_facts: list[str]
    ) -> JudgeVerdict | None:
        response = await self._client.aio.models.generate_content(
            model=self.model,
            contents=build_judge_prompt(question, chunks, answer, expected_facts),
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0,
                response_mime_type="application/json",
                response_schema=JudgeVerdict,
            ),
        )
        return response.parsed if isinstance(response.parsed, JudgeVerdict) else None
