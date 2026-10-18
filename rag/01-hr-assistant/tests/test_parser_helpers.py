"""Pure helpers of the Docling parser (no PDF conversion, no models loaded)."""
import pytest

from hr_assistant.ingestion.docling_parser import compact_table, heading_level


@pytest.mark.parametrize(
    ("text", "is_first", "expected"),
    [
        ("Leave Policy 2026", True, 0),  # unnumbered first heading = document title
        ("Appendix", False, 1),  # unnumbered later heading = top-level section
        ("3. Leave Entitlements", False, 1),
        ("3.1 Casual Leave", False, 2),
        ("3.1.2 Accrual", False, 3),
        ("2026 Holiday List", False, 1),  # a year followed by a space looks numbered: documented limitation
    ],
)
def test_heading_level_is_inferred_from_section_numbers(text, is_first, expected):
    assert heading_level(text, is_first) == expected


def test_compact_table_strips_padding_but_keeps_content():
    padded = (
        "| Grade   | Example titles      |   Minimum |\n"
        "|---------|---------------------|-----------|\n"
        "| L1      | Associate Engineer  |       4.0 |"
    )

    assert compact_table(padded) == (
        "| Grade | Example titles | Minimum |\n"
        "| --- | --- | --- |\n"
        "| L1 | Associate Engineer | 4.0 |"
    )
