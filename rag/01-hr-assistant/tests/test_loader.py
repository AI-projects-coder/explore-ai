"""The manifest loader: default-deny, validation, and stable document identity."""
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from hr_assistant.ingestion.loader import LocalManifestLoader

FAKE_PDF = b"%PDF-1.4 fake bytes"


def make_folder(tmp_path: Path, manifest: str, files: list[str]) -> Path:
    (tmp_path / "manifest.yaml").write_text(manifest)
    for name in files:
        (tmp_path / name).write_bytes(FAKE_PDF)
    return tmp_path


def test_loads_listed_files_with_stable_uri_hash_and_sorted_groups(tmp_path):
    folder = make_folder(
        tmp_path,
        "documents:\n  - file: a.pdf\n    title: A\n    allowed_groups: [manager, hr, manager]\n",
        ["a.pdf"],
    )

    docs, problems = LocalManifestLoader(folder).load()

    assert problems == []
    assert len(docs) == 1
    assert docs[0].source_uri == "local://a.pdf"  # not the absolute path: same on every machine
    assert docs[0].content_hash == hashlib.sha256(FAKE_PDF).hexdigest()
    assert docs[0].allowed_groups == ["hr", "manager"]  # de-duplicated and sorted


def test_pdf_not_in_manifest_is_skipped_and_reported(tmp_path):
    folder = make_folder(
        tmp_path,
        "documents:\n  - file: a.pdf\n    title: A\n    allowed_groups: [employee]\n",
        ["a.pdf", "draft.pdf"],
    )

    docs, problems = LocalManifestLoader(folder).load()

    assert [d.source_uri for d in docs] == ["local://a.pdf"]
    assert problems == ["draft.pdf: not in manifest, skipped (default-deny)"]


def test_listed_file_that_does_not_exist_is_reported(tmp_path):
    folder = make_folder(
        tmp_path,
        "documents:\n  - file: gone.pdf\n    title: Gone\n    allowed_groups: [employee]\n",
        [],
    )

    docs, problems = LocalManifestLoader(folder).load()

    assert docs == []
    assert problems == ["gone.pdf: listed in manifest but file not found"]


@pytest.mark.parametrize(
    ("entry", "error"),
    [
        ("  - file: ../../.env\n    title: X\n    allowed_groups: [employee]\n", "plain file name"),
        ("  - file: a.pdf\n    title: X\n    allowed_groups: []\n", "at least 1 item"),
        ("  - file: a.pdf\n    title: ''\n    allowed_groups: [employee]\n", "at least 1 character"),
        (
            "  - file: a.pdf\n    title: X\n    allowed_groups: [employee]\n"
            "  - file: a.pdf\n    title: Y\n    allowed_groups: [hr]\n",
            "duplicate entries",
        ),
    ],
    ids=["path-traversal", "empty-groups", "empty-title", "duplicate-file"],
)
def test_invalid_manifest_stops_the_whole_load(tmp_path, entry, error):
    folder = make_folder(tmp_path, "documents:\n" + entry, ["a.pdf"])

    with pytest.raises(ValidationError, match=error):
        LocalManifestLoader(folder).load()


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest.yaml"):
        LocalManifestLoader(tmp_path).load()
