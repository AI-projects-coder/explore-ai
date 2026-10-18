"""The public API contract: what clients send and receive. Internal details never appear here."""
from pydantic import BaseModel, ConfigDict, Field, field_validator

from hr_assistant.generation.answer import Answer, Citation

MAX_QUESTION_CHARS = 1000


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # e.g. a client-sent "groups" field is rejected, not ignored

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question must not be blank")
        return v.strip()


class CitationOut(BaseModel):
    number: int
    title: str
    section: str
    pages: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[CitationOut]

    @classmethod
    def from_answer(cls, answer: Answer) -> "AskResponse":
        return cls(
            answer=answer.text,
            answerable=answer.answerable,
            citations=[_citation_out(c) for c in answer.citations],
        )


def _citation_out(c: Citation) -> CitationOut:
    parts = c.heading_path.split(" > ")
    section = " > ".join(parts[1:]) or c.title  # heading_path starts with the title
    if c.page_start is None:
        pages = ""
    elif c.page_start == c.page_end:
        pages = f"p. {c.page_start}"
    else:
        pages = f"pp. {c.page_start}-{c.page_end}"
    return CitationOut(number=c.number, title=c.title, section=section, pages=pages)


class DevTokenRequest(BaseModel):
    user_id: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
