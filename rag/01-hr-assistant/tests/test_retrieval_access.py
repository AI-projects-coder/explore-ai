"""Access control at retrieval time: a user only ever receives chunks their groups allow."""
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from hr_assistant.core.auth import User
from hr_assistant.retrieval.search import Retriever
from tests.support import vector_for

INTERN = User("intern_ravi", frozenset({"employee"}))
MANAGER = User("manager_priya", frozenset({"employee", "manager"}))
HR = User("hr_anita", frozenset({"employee", "hr"}))


async def test_intern_never_sees_salary_even_when_it_is_the_best_match(retriever, embedder):
    # Arrange: make a salary chunk the PERFECT match (similarity 1.0) for the question.
    embedder.query_vector = vector_for("salary-grades")

    # Act
    results = await retriever.search("What is the maximum salary for L4?", INTERN)

    # Assert: no salary chunk, and only documents an employee may read.
    titles = {r.title for r in results}
    assert "Salary Bands" not in titles
    assert titles == {"Leave Policy"}


async def test_manager_sees_salary(retriever, embedder):
    embedder.query_vector = vector_for("salary-grades")

    results = await retriever.search("What is the maximum salary for L4?", MANAGER)

    assert results[0].title == "Salary Bands"
    assert results[0].score == 1.0


async def test_hr_sees_salary_but_not_performance_guide(retriever, embedder):
    embedder.query_vector = vector_for("perf-calibration")  # the manager-only chunk is the best match

    results = await retriever.search("How do I calibrate ratings?", HR)

    titles = {r.title for r in results}
    assert "Performance Guide" not in titles  # manager-only
    assert "Salary Bands" in titles  # hr is allowed here


async def test_user_without_groups_gets_nothing_and_embedder_is_not_called(retriever, embedder):
    results = await retriever.search("anything", User("nobody", frozenset()))

    assert results == []
    assert embedder.calls == 0  # no paid API call, no database query


async def test_user_whose_groups_match_no_document_gets_nothing(retriever, embedder):
    results = await retriever.search("leave policy", User("vendor", frozenset({"contractor"})))

    assert results == []
    assert embedder.calls == 1  # the search ran; the SQL filter returned nothing


async def test_deleted_document_is_never_returned(retriever, embedder):
    embedder.query_vector = vector_for("old-travel")  # chunk of a DELETED document is the best match

    results = await retriever.search("travel policy", INTERN)

    assert "Old Travel Policy" not in {r.title for r in results}


async def test_chunks_from_a_different_embedding_model_are_ignored(retriever, embedder):
    embedder.model = "some-newer-embedding-model"  # e.g. mid blue/green re-index

    results = await retriever.search("leave", INTERN)

    assert results == []


async def test_results_are_sorted_best_first(retriever, embedder):
    embedder.query_vector = vector_for("leave-sick")

    results = await retriever.search("sick leave", INTERN)

    assert results[0].heading_path == "Leave Policy > leave-sick"
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


async def test_min_score_drops_weak_matches(database, embedder):
    embedder.query_vector = vector_for("leave-casual")
    engine = create_async_engine(database)
    strict = Retriever(embedder=embedder, session_factory=async_sessionmaker(engine), min_score=0.5)
    try:
        results = await strict.search("casual leave", INTERN)
    finally:
        await engine.dispose()

    assert [r.heading_path for r in results] == ["Leave Policy > leave-casual"]  # the 0.0 matches are gone
