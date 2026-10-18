import asyncio
import logging

from hr_assistant.core.auth import User
from hr_assistant.core.config import get_settings
from hr_assistant.db.session import get_async_session_factory
from hr_assistant.embeddings.vertex import VertexEmbedder
from hr_assistant.generation.answer import AnswerService
from hr_assistant.generation.llm import GeminiLLM
from hr_assistant.retrieval.search import Retriever

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)


async def main() -> None:
    s = get_settings()
    service = AnswerService(
        retriever=Retriever(
            embedder=VertexEmbedder.from_settings(s),
            session_factory=get_async_session_factory(),
            top_k=s.retrieval_top_k,
            min_score=s.retrieval_min_score,
            ef_search=s.hnsw_ef_search,
        ),
        llm=GeminiLLM.from_settings(s),
        no_answer_message=s.no_answer_message,
    )
    intern = User("intern_ravi", frozenset({"employee"}))
    for question in [
        "How many casual leaves can I carry forward?",
        "Can I work from Dubai for a month?",
        "What is the maximum salary for an L4 lead engineer?",
    ]:
        a = await service.answer(question, intern)
        print(f"\nQ: {question}\nA: {a.text}")
        for c in a.citations:
            print(f"   [{c.number}] {c.heading_path} (p{c.page_start})")


asyncio.run(main())
