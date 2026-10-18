import math

from hr_assistant.core.config import get_settings
from hr_assistant.embeddings.vertex import VertexEmbedder

emb = VertexEmbedder.from_settings(get_settings())


def cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))


policy = emb.embed_documents(["Employees may work from another city in India for up to 30 days a year."]).vectors[0]
for question in [
    "Can I work from my hometown for two weeks?",
    "Can I work from Dubai?",
    "What is the dress code?",
]:
    print(f"{cos(policy, emb.embed_query(question)):.3f}  {question}")
