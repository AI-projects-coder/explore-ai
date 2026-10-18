"""Test helpers shared by several test files: seed data, deterministic vectors, a fake embedder."""
from hr_assistant.db.models import EMBEDDING_DIM

TEST_EMBEDDING_MODEL = "test-embedding-model"

# The seed data. Every chunk gets its own direction in vector space (see vector_for),
# so a test can make ANY chunk the perfect match for a query.
SEED_DOCUMENTS = [
    {"title": "Leave Policy", "groups": ["employee"], "status": "active",
     "chunks": ["leave-casual", "leave-sick"]},
    {"title": "Salary Bands", "groups": ["hr", "manager"], "status": "active",
     "chunks": ["salary-grades", "salary-increments"]},
    {"title": "Performance Guide", "groups": ["manager"], "status": "active",
     "chunks": ["perf-calibration"]},
    # A deleted document whose chunks were (wrongly) left behind: tests the "defense in depth" filter.
    {"title": "Old Travel Policy", "groups": ["employee"], "status": "deleted",
     "chunks": ["old-travel"]},
]
_CHUNK_NAMES = [name for doc in SEED_DOCUMENTS for name in doc["chunks"]]


def vector_for(chunk_name: str) -> list[float]:
    """A one-hot vector: cosine similarity is 1.0 with this chunk and 0.0 with every other chunk."""
    vector = [0.0] * EMBEDDING_DIM
    vector[_CHUNK_NAMES.index(chunk_name)] = 1.0
    return vector


class FakeEmbedder:
    """Stands in for Vertex AI: no network, no cost, same answer every time."""

    model = TEST_EMBEDDING_MODEL

    def __init__(self) -> None:
        self.query_vector = vector_for("leave-casual")
        self.calls = 0

    async def embed_query_async(self, text: str) -> list[float]:
        self.calls += 1
        return self.query_vector
