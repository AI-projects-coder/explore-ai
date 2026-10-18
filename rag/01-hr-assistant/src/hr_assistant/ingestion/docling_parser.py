import io
import re

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel, TableItem, TextItem
from docling_core.types.io import DocumentStream

from hr_assistant.ingestion.models import Block, BlockType, ParsedDocument

# "3. Leave Entitlements" -> "3", "3.1 Casual Leave" -> "3.1"
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s")
_HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}
_TEXT_LABELS = {DocItemLabel.TEXT, DocItemLabel.PARAGRAPH, DocItemLabel.LIST_ITEM}


def heading_level(text: str, is_first_heading: bool) -> int:
    """PDFs don't store heading depth, so infer it from the section number."""
    match = _NUMBERED_HEADING.match(text)
    if match:
        return match.group(1).count(".") + 1
    return 0 if is_first_heading else 1


class DoclingParser:
    def __init__(self) -> None:
        # Loads layout/table models: slow, so do it once and reuse for every document.
        self._converter = DocumentConverter()

    def parse(self, data: bytes, filename: str) -> ParsedDocument:
        result = self._converter.convert(
            DocumentStream(name=filename, stream=io.BytesIO(data)), raises_on_error=False
        )
        if result.status != ConversionStatus.SUCCESS:
            raise ValueError(f"{filename}: docling conversion {result.status.value}")
        doc = result.document

        blocks: list[Block] = []
        seen_heading = False
        # iterate_items() yields body content only; page headers/footers ("furniture") are excluded.
        for item, _ in doc.iterate_items():
            if not item.prov:
                continue
            page = item.prov[0].page_no

            if isinstance(item, TableItem):
                blocks.append(Block(BlockType.TABLE, item.export_to_markdown(doc=doc), page))
            elif isinstance(item, TextItem) and item.text.strip():
                text = item.text.strip()
                if item.label in _HEADING_LABELS:
                    blocks.append(Block(BlockType.HEADING, text, page, heading_level(text, not seen_heading)))
                    seen_heading = True
                elif item.label in _TEXT_LABELS:
                    blocks.append(Block(BlockType.TEXT, text, page))
                # other labels (captions, footnotes, ...) are ignored for now

        if not blocks:
            raise ValueError(f"{filename}: no text extracted (scanned PDF without OCR?)")
        return ParsedDocument(blocks=blocks, page_count=len(doc.pages))
