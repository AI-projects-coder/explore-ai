import asyncio
import logging

from hr_assistant.core.auth import User
from hr_assistant.core.config import get_settings
from hr_assistant.db.session import get_async_session_factory
from hr_assistant.embeddings.vertex import VertexEmbedder
from hr_assistant.retrieval.search import Retriever

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request HTTP logs


async def main() -> None:
    settings = get_settings()
    retriever = Retriever(
        embedder=VertexEmbedder.from_settings(settings),
        session_factory=get_async_session_factory(),
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
        ef_search=settings.hnsw_ef_search,
    )
    for user in [User("intern_ravi", frozenset({"employee"})), User("manager_priya", frozenset({"employee", "manager"}))]:
        print(f"\n{user.id}")
        for c in await retriever.search("What is the maximum salary for an L4 lead engineer?", user):
            print(f"  {c.score:.3f}  {c.title} > {c.heading_path.split(' > ')[-1]}  (p{c.page_start}-{c.page_end})")


asyncio.run(main())
