from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class BlockType(StrEnum):
    HEADING = "heading"
    TEXT = "text"
    TABLE = "table"


@dataclass(frozen=True)
class SourceDocument:
    source_type: str          # "pdf"
    source_uri: str           # stable ID, e.g. "local://leave-policy-2026.pdf"
    title: str
    allowed_groups: list[str]
    data: bytes
    content_hash: str         # sha256 of data
    source_version: str | None = None


@dataclass(frozen=True)
class Block:
    type: BlockType
    text: str
    page: int
    level: int | None = None  # only for headings: 0 = title, 1 = "3.", 2 = "3.1"


@dataclass(frozen=True)
class ParsedDocument:
    blocks: list[Block]
    page_count: int


class Parser(Protocol):
    def parse(self, data: bytes, filename: str) -> ParsedDocument: ...
