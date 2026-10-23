# Agent #1: Financial Research Analyst

An autonomous research agent: "Analyze Apple's last 4 quarters" → plan → research (SEC XBRL + 8-K) → calculate → write.

Built with LangGraph (orchestration only) and google-genai (Vertex AI) inside nodes.

Status: v0 (local, CLI, in-memory state)

## Usage (v1)

```
docker compose up -d                 # local Postgres on port 5434
uv run research-analyst init-db      # once: LangGraph checkpoint tables + our runs table
uv run research-analyst run "Analyze Apple's last 4 quarters"
uv run research-analyst resume run-1a2b3c4d   # after a crash or "decide later"
uv run research-analyst list
uv run research-analyst show run-1a2b3c4d
```
