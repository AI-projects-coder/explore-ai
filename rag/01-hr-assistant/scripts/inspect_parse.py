import sys
from pathlib import Path

from hr_assistant.ingestion.docling_parser import DoclingParser
from hr_assistant.ingestion.loader import LocalManifestLoader
from hr_assistant.ingestion.chunker import group_into_sections


from hr_assistant.ingestion.chunker import count_tokens, group_into_sections, split_section
from hr_assistant.ingestion.chunker import chunk_document

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




for s in group_into_sections(parsed.blocks, d.title):
    pages = sorted({b.page for b in s.blocks})
    print(f"p{pages[0]}-{pages[-1]}  {len(s.body)} body  {' > '.join(s.heading_path)}")


for s in group_into_sections(parsed.blocks, d.title):
    pieces = split_section(s, max_tokens=400, overlap_tokens=50)
    for i, pc in enumerate(pieces, 1):
        toks = sum(count_tokens(b.text) for b in pc.body)
        print(f"{toks:4d} tok  piece {i}/{len(pieces)}  {pc.heading_path[-1]}")


chunks = chunk_document(parsed, d.title, max_tokens=400, min_tokens=50, overlap_tokens=50)
print(f"{len(chunks)} chunks")
for c in chunks:
    print(f"#{c.chunk_index} {c.token_count:4d}tok p{c.page_start}-{c.page_end} | {c.heading_path}")
