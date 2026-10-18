"""The chunker: heading paths, page ranges, splitting, table handling, merging, hashing."""
from hr_assistant.ingestion.chunker import (
    chunk_document,
    count_tokens,
    group_into_sections,
    merge_small_sections,
    split_section,
    split_table,
)
from hr_assistant.ingestion.models import Block, BlockType, ParsedDocument


def heading(text: str, level: int, page: int = 1) -> Block:
    return Block(BlockType.HEADING, text, page, level)


def para(text: str, page: int = 1) -> Block:
    return Block(BlockType.TEXT, text, page)


def test_heading_path_follows_the_document_hierarchy():
    blocks = [
        heading("Leave Policy", 0),
        para("Version 1."),
        heading("3. Leave Entitlements", 1),
        para("Intro."),
        heading("3.1 Casual Leave", 2),
        para("Casual text."),
        heading("4. Probation", 1),  # back up one level: "3." must be popped
        para("Probation text."),
    ]

    paths = [s.heading_path for s in group_into_sections(blocks, "Leave Policy 2026")]

    assert paths == [
        ["Leave Policy 2026"],  # manifest title, not the PDF's own L0 heading
        ["Leave Policy 2026", "3. Leave Entitlements"],
        ["Leave Policy 2026", "3. Leave Entitlements", "3.1 Casual Leave"],
        ["Leave Policy 2026", "4. Probation"],
    ]


def test_heading_only_section_is_dropped_but_kept_in_child_paths():
    blocks = [heading("3. Entitlements", 1), heading("3.1 Casual", 2), para("text")]

    sections = group_into_sections(blocks, "Doc")

    assert [s.heading_path for s in sections] == [["Doc", "3. Entitlements", "3.1 Casual"]]


def test_section_spanning_pages_keeps_the_heading_page():
    blocks = [heading("3.4 Parental Leave", 2, page=1), para("Body on next page.", page=2)]

    chunks = chunk_document(ParsedDocument(blocks, 2), "Doc", max_tokens=400, min_tokens=0, overlap_tokens=0)

    assert (chunks[0].page_start, chunks[0].page_end) == (1, 2)


def test_long_section_is_split_and_next_piece_starts_with_overlap():
    sentences = [f"Sentence number {i} explains one rule of the policy in detail." for i in range(30)]
    section = group_into_sections([heading("2. Respect", 1), para(" ".join(sentences))], "Doc")[0]

    pieces = split_section(section, max_tokens=100, overlap_tokens=20)

    assert len(pieces) > 1
    for piece in pieces:
        assert sum(count_tokens(b.text) for b in piece.body) <= 100 + 20
        assert piece.heading_path == ["Doc", "2. Respect"]
    last_sentence_of_first = pieces[0].body[-1].text.split(". ")[-1]
    assert pieces[1].body[0].text.endswith(last_sentence_of_first)  # overlap carried over


def test_big_table_is_split_between_rows_and_every_piece_repeats_the_header():
    header = "| Grade | Min | Max |\n| --- | --- | --- |"
    rows = [f"| L{i} | {i}.0 | {i + 5}.0 |" for i in range(40)]

    pieces = split_table("\n".join([header, *rows]), max_tokens=60)

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith(header)
        assert count_tokens(piece) <= 60
    all_rows = [line for p in pieces for line in p.splitlines()[2:]]
    assert all_rows == rows  # every row exactly once, in order


def test_small_sections_merge_under_their_common_parent():
    blocks = [
        heading("WFH Policy", 0),
        heading("1. Purpose", 1),
        para("Short."),
        heading("2. Eligibility", 1),
        para("Also short."),
    ]
    sections = group_into_sections(blocks, "WFH Policy")

    merged = merge_small_sections(sections, min_tokens=50, max_tokens=400)

    assert len(merged) == 1
    assert merged[0].heading_path == ["WFH Policy"]  # shared parent
    assert [b.text for b in merged[0].blocks if b.type == BlockType.HEADING][-2:] == ["1. Purpose", "2. Eligibility"]


def test_merging_never_exceeds_max_tokens():
    blocks = [heading("Doc", 0)]
    for i in range(10):
        blocks += [heading(f"{i + 1}. Section", 1), para("word " * 40)]

    merged = merge_small_sections(group_into_sections(blocks, "Doc"), min_tokens=100, max_tokens=150)

    assert all(sum(count_tokens(b.text) for b in s.body) <= 150 for s in merged)


def test_hash_changes_when_title_changes_so_the_chunk_is_re_embedded():
    parsed = ParsedDocument([heading("1. Rule", 1), para("Same body text.")], 1)

    a = chunk_document(parsed, "Old Title", max_tokens=400, min_tokens=0, overlap_tokens=0)
    b = chunk_document(parsed, "New Title", max_tokens=400, min_tokens=0, overlap_tokens=0)

    assert a[0].content == b[0].content
    assert a[0].content_hash != b[0].content_hash


def test_chunking_is_deterministic():
    parsed = ParsedDocument([heading("1. Rule", 1), para("Body. " * 200)], 1)

    first = chunk_document(parsed, "Doc", max_tokens=100, min_tokens=20, overlap_tokens=20)
    second = chunk_document(parsed, "Doc", max_tokens=100, min_tokens=20, overlap_tokens=20)

    assert [c.content_hash for c in first] == [c.content_hash for c in second]
