import logging

from hr_assistant.core.config import get_settings
from hr_assistant.embeddings.vertex import VertexEmbedder

logging.basicConfig(level=logging.INFO)
emb = VertexEmbedder.from_settings(get_settings())

result = emb.embed_documents(["Casual leave can be carried forward up to 4 days.", "Internet is reimbursed up to INR 1,000."])
print(len(result.vectors), "vectors,", len(result.vectors[0]), "dims,", result.token_count, "tokens")

q = emb.embed_query("How many casual leaves can I carry forward?")
print("query dims:", len(q))
