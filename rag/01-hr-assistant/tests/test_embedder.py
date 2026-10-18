"""VertexEmbedder with the Google client mocked: batching, retries, and safety checks. No network."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from google.genai import errors
from tenacity import wait_none

from hr_assistant.core.config import get_settings
from hr_assistant.db.models import EMBEDDING_DIM
from hr_assistant.embeddings.vertex import VertexEmbedder


def fake_response(n: int, *, truncated: bool = False, dims: int = EMBEDDING_DIM):
    embedding = SimpleNamespace(
        values=[0.1] * dims, statistics=SimpleNamespace(truncated=truncated, token_count=7)
    )
    return SimpleNamespace(embeddings=[embedding] * n)


@pytest.fixture
def embedder(monkeypatch):
    # No real waiting between retries in tests.
    monkeypatch.setattr(VertexEmbedder._embed.retry, "wait", wait_none())
    with patch("hr_assistant.embeddings.vertex.genai.Client") as client_cls:
        emb = VertexEmbedder(project="p", location="l", model="m", batch_size=3)
        emb.api = client_cls.return_value.models.embed_content  # the mocked API call
        yield emb


def test_texts_are_sent_in_batches_of_batch_size(embedder):
    embedder.api.side_effect = lambda **kw: fake_response(len(kw["contents"]))

    result = embedder.embed_documents([f"t{i}" for i in range(7)])

    assert [len(call.kwargs["contents"]) for call in embedder.api.call_args_list] == [3, 3, 1]
    assert len(result.vectors) == 7
    assert result.token_count == 7 * 7


def test_temporary_503_is_retried_then_succeeds(embedder):
    unavailable = errors.ServerError(503, {"error": {"message": "unavailable"}})
    embedder.api.side_effect = [unavailable, unavailable, fake_response(1)]

    result = embedder.embed_documents(["text"])

    assert embedder.api.call_count == 3
    assert len(result.vectors) == 1


def test_bad_request_400_is_not_retried(embedder):
    embedder.api.side_effect = errors.ClientError(400, {"error": {"message": "bad request"}})

    with pytest.raises(errors.ClientError):
        embedder.embed_documents(["text"])

    assert embedder.api.call_count == 1


def test_gives_up_after_five_attempts(embedder):
    embedder.api.side_effect = errors.ServerError(503, {"error": {"message": "down"}})

    with pytest.raises(errors.ServerError):
        embedder.embed_documents(["text"])

    assert embedder.api.call_count == 5


def test_truncated_text_raises_instead_of_storing_a_partial_vector(embedder):
    embedder.api.return_value = fake_response(1, truncated=True)

    with pytest.raises(ValueError, match="truncated"):
        embedder.embed_documents(["very long text"])


def test_missing_vector_in_response_raises(embedder):
    embedder.api.return_value = fake_response(1)  # sent 2 texts, got 1 vector back

    with pytest.raises(ValueError, match="sent 2 texts, got 1"):
        embedder.embed_documents(["a", "b"])


def test_wrong_dimension_raises(embedder):
    embedder.api.return_value = fake_response(1, dims=768)

    with pytest.raises(ValueError, match="expected 1536 dimensions"):
        embedder.embed_documents(["text"])


def test_config_dimension_mismatch_refuses_to_start():
    settings = get_settings().model_copy(update={"embedding_dim": 768})

    with patch("hr_assistant.embeddings.vertex.genai.Client"), pytest.raises(ValueError, match="vector\\(1536\\)"):
        VertexEmbedder.from_settings(settings)


def test_empty_vector_raises():
    with pytest.raises(ValueError, match="expected 1536 dimensions, got 0"):
        VertexEmbedder._checked(None)
