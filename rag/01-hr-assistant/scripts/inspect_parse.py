import sys
from pathlib import Path

from hr_assistant.ingestion.docling_parser import DoclingParser
from hr_assistant.ingestion.loader import LocalManifestLoader

docs, problems = LocalManifestLoader(Path(sys.argv[1])).load()
print(f"loaded {len(docs)} documents")
for p in problems:
    print("PROBLEM:", p)

parser = DoclingParser()
for d in docs:
    if d.source_uri.endswith(sys.argv[2]):
        print(f"\n{d.title}  groups={d.allowed_groups}  hash={d.content_hash[:12]}")
        parsed = parser.parse(d.data, d.source_uri)
        print(f"pages={parsed.page_count} blocks={len(parsed.blocks)}")
        for b in parsed.blocks:
            lvl = f"L{b.level}" if b.level is not None else "  "
            print(f"p{b.page} {b.type:8} {lvl} | {b.text.splitlines()[0][:70]}")
