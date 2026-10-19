"""Show a golden case next to the source text it depends on. Usage: uv run python scripts/review_case.py <case-id>"""
import sys

import yaml
from sqlalchemy import text

from hr_assistant.core.auth import DEV_USERS
from hr_assistant.db.session import get_engine

case = next(c for c in yaml.safe_load(open("evals/golden.yaml"))["cases"] if c["id"] == sys.argv[1])
print(f"QUESTION : {case['question']}")
print(f"USER     : {case['user']}  groups={sorted(DEV_USERS[case['user']])}")
print(f"EXPECTED : outcome={case['expected_outcome']}  must_contain={case.get('must_contain', [])}  "
      f"must_not_contain={case.get('must_not_contain', [])}")

with get_engine().connect() as conn:
    rows = conn.execute(text(
        "SELECT d.title, d.allowed_groups, c.heading_path, c.content "
        "FROM chunks c JOIN documents d ON d.id = c.document_id"
    )).all()

for section in case.get("expected_sections", []):
    for alt in section.split("|"):
        for r in rows:
            if alt in r.heading_path or f"## {alt}" in r.content:
                print(f"\n--- {r.heading_path}   (document groups: {r.allowed_groups})")
                print(r.content)

# Where ELSE does each expected fact appear? (question 5: is the expected section the only place?)
for fact in case.get("must_contain", []):
    for alt in fact.split("|"):
        others = [r.heading_path for r in rows if alt.lower() in r.content.lower().replace(",", "")]
        print(f"\n'{alt}' also appears in: {others}")
