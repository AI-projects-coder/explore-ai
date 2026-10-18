import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

from hr_assistant.ingestion.models import SourceDocument

MANIFEST_NAME = "manifest.yaml"
SUPPORTED_SUFFIXES = {".pdf"}


class ManifestEntry(BaseModel):
    file: str
    title: str = Field(min_length=1)
    allowed_groups: list[str] = Field(min_length=1)

    @field_validator("file")
    @classmethod
    def _plain_filename(cls, v: str) -> str:
        # Reject "../secret.pdf" or "sub/dir.pdf": the manifest may only point inside its own folder.
        if Path(v).name != v:
            raise ValueError(f"must be a plain file name, got {v!r}")
        return v


class Manifest(BaseModel):
    documents: list[ManifestEntry]

    @field_validator("documents")
    @classmethod
    def _no_duplicates(cls, v: list[ManifestEntry]) -> list[ManifestEntry]:
        names = [e.file for e in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate entries: {sorted(dupes)}")
        return v


class LocalManifestLoader:
    """Loads documents from a local folder; manifest.yaml supplies titles and access groups."""
    source_type = "pdf"
    uri_prefix = "local://"
    def __init__(self, root: Path) -> None:
        self.root = root

    def load(self) -> tuple[list[SourceDocument], list[str]]:
        manifest = self._read_manifest()
        documents: list[SourceDocument] = []
        problems: list[str] = []

        for entry in manifest.documents:
            path = self.root / entry.file
            if not path.is_file():
                problems.append(f"{entry.file}: listed in manifest but file not found")
                continue
            data = path.read_bytes()
            documents.append(
                SourceDocument(
                    source_type="pdf",
                    source_uri=f"local://{entry.file}",
                    title=entry.title,
                    allowed_groups=sorted(set(entry.allowed_groups)),
                    data=data,
                    content_hash=hashlib.sha256(data).hexdigest(),
                )
            )

        listed = {e.file for e in manifest.documents}
        for path in sorted(self.root.iterdir()):
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.name not in listed:
                problems.append(f"{path.name}: not in manifest, skipped (default-deny)")

        return documents, problems

    def _read_manifest(self) -> Manifest:
        path = self.root / MANIFEST_NAME
        if not path.is_file():
            raise FileNotFoundError(f"{MANIFEST_NAME} not found in {self.root}")
        return Manifest.model_validate(yaml.safe_load(path.read_text()))
