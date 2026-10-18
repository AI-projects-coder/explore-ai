from dataclasses import dataclass, field, replace

import math
import re
import hashlib


from hr_assistant.ingestion.models import Block, BlockType, ChunkDraft, ParsedDocument





@dataclass
class Section:
    """A heading plus the blocks under it (before the next heading)."""

    heading_path: list[str]
    blocks: list[Block] = field(default_factory=list)

    @property
    def body(self) -> list[Block]:
        return [b for b in self.blocks if b.type != BlockType.HEADING]


def group_into_sections(blocks: list[Block], title: str) -> list[Section]:
    """Stage 1: walk the blocks, tracking a heading stack, and cut a section at every heading.

    The path always starts with the manifest title; the document's own L0 heading is not repeated.
    """
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading text) for levels >= 1
    current = Section(heading_path=[title])

    for block in blocks:
        if block.type != BlockType.HEADING:
            current.blocks.append(block)
            continue

        if current.body:  # a heading with no body (e.g. "3." directly followed by "3.1") is dropped
            sections.append(current)

        level = block.level or 0
        while stack and stack[-1][0] >= level:  # pop siblings and deeper headings
            stack.pop()
        if level >= 1:
            stack.append((level, block.text))
        current = Section(heading_path=[title, *(text for _, text in stack)], blocks=[block])

    if current.body:
        sections.append(current)
    return sections

# ---------------------------------------------------------------- Stage 2: split big sections

def count_tokens(text: str) -> int:
    """Approximate ~4 chars/token. Measured ~20% high for Gemini, which errs on the safe side."""
    return max(1, math.ceil(len(text) / 4))


# A sentence ends at . ! or ? followed by whitespace and an uppercase letter, digit, or quote/bracket.
# "1.5 days" is not split (no space after the dot); "e.g. the" is not split (lowercase follows).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def split_table(markdown: str, max_tokens: int) -> list[str]:
    """Split a markdown table between rows, repeating the header row + separator in every piece."""
    lines = markdown.strip().splitlines()
    header, rows = lines[:2], lines[2:]
    pieces: list[str] = []
    current: list[str] = []
    for row in rows:
        if current and count_tokens("\n".join([*header, *current, row])) > max_tokens:
            pieces.append("\n".join([*header, *current]))
            current = []
        current.append(row)
    if current:
        pieces.append("\n".join([*header, *current]))
    return pieces


def _to_units(block: Block, max_tokens: int) -> list[Block]:
    """Break one block into units that each fit in max_tokens (tables by rows, text by sentences)."""
    if count_tokens(block.text) <= max_tokens:
        return [block]
    if block.type == BlockType.TABLE:
        return [replace(block, text=t) for t in split_table(block.text, max_tokens)]
    return [replace(block, text=s) for s in split_sentences(block.text)]


def _overlap_tail(blocks: list[Block], overlap_tokens: int) -> list[Block]:
    """The last sentences of the previous piece (up to overlap_tokens). Text only, never tables."""
    last = blocks[-1]
    if last.type != BlockType.TEXT:
        return []
    tail: list[str] = []
    for sentence in reversed(split_sentences(last.text)):
        if count_tokens(" ".join([sentence, *tail])) > overlap_tokens:
            break
        tail.insert(0, sentence)
    return [replace(last, text=" ".join(tail))] if tail else []


def split_section(section: Section, max_tokens: int, overlap_tokens: int) -> list[Section]:
    """Stage 2: split a section bigger than max_tokens into pieces with the same heading_path."""
    if sum(count_tokens(b.text) for b in section.body) <= max_tokens:
        return [section]

    headings = [b for b in section.blocks if b.type == BlockType.HEADING]
    units = [unit for block in section.body for unit in _to_units(block, max_tokens)]

    pieces: list[Section] = []
    current: list[Block] = list(headings)  # the heading stays with the first piece (its page counts)
    size = 0
    for unit in units:
        unit_size = count_tokens(unit.text)
        if size and size + unit_size > max_tokens:
            pieces.append(Section(section.heading_path, current))
            current = _overlap_tail(current, overlap_tokens)
            size = sum(count_tokens(b.text) for b in current)
        current.append(unit)
        size += unit_size
    pieces.append(Section(section.heading_path, current))
    return pieces


# ---------------------------------------------------------------- Stage 3: merge small sections

def _size(section: Section) -> int:
    return sum(count_tokens(b.text) for b in section.body)


def _common_prefix(a: list[str], b: list[str]) -> list[str]:
    prefix: list[str] = []
    for x, y in zip(a, b):
        if x != y:
            break
        prefix.append(x)
    return prefix


def _combine(a: Section, b: Section) -> Section:
    """Join two sections; the path becomes their shared parent, sub-headings stay in the blocks."""
    return Section(_common_prefix(a.heading_path, b.heading_path), [*a.blocks, *b.blocks])


def merge_small_sections(sections: list[Section], min_tokens: int, max_tokens: int) -> list[Section]:
    """Stage 3: merge consecutive sections smaller than min_tokens, never exceeding max_tokens."""
    merged: list[Section] = []
    for section in sections:
        prev = merged[-1] if merged else None
        if prev and _size(prev) < min_tokens and _size(prev) + _size(section) <= max_tokens:
            merged[-1] = _combine(prev, section)
        else:
            merged.append(section)

    # A small section left at the end joins the previous chunk if it fits.
    if len(merged) >= 2 and _size(merged[-1]) < min_tokens and _size(merged[-2]) + _size(merged[-1]) <= max_tokens:
        merged[-2:] = [_combine(merged[-2], merged[-1])]
    return merged


# ---------------------------------------------------------------- Stage 4: build chunks

def _render(blocks: list[Block]) -> str:
    return "\n\n".join(f"## {b.text}" if b.type == BlockType.HEADING else b.text for b in blocks)


def chunk_document(
    parsed: ParsedDocument, title: str, *, max_tokens: int, min_tokens: int, overlap_tokens: int
) -> list[ChunkDraft]:
    """Blocks -> sections -> split big -> merge small -> ChunkDrafts."""
    sections = group_into_sections(parsed.blocks, title)
    pieces = [p for s in sections for p in split_section(s, max_tokens, overlap_tokens)]
    merged = merge_small_sections(pieces, min_tokens, max_tokens)

    chunks: list[ChunkDraft] = []
    for index, section in enumerate(merged):
        heading_path = " > ".join(section.heading_path)
        content = _render(section.blocks)
        embed_text = f"{heading_path}\n\n{content}"
        pages = [b.page for b in section.blocks]
        chunks.append(
            ChunkDraft(
                chunk_index=index,
                heading_path=heading_path,
                content=content,
                embed_text=embed_text,
                content_hash=hashlib.sha256(embed_text.encode()).hexdigest(),
                page_start=min(pages),
                page_end=max(pages),
                token_count=count_tokens(embed_text),
            )
        )
    return chunks
