from app.documents import clean_extracted_text, extract_pdf_text


def test_clean_extracted_text_handles_none() -> None:
    assert clean_extracted_text(None) == ""


def test_clean_extracted_text_handles_empty_string() -> None:
    assert clean_extracted_text("") == ""


def test_clean_extracted_text_normalizes_spaces_tabs_and_safe_artifacts() -> None:
    assert clean_extracted_text("  Alpha   beta\t\tgamma\x00 delta\x0c next  ") == (
        "Alpha beta gamma delta\nnext"
    )


def test_clean_extracted_text_collapses_excessive_blank_lines() -> None:
    assert clean_extracted_text("First\n\n\n\n\nSecond") == "First\n\nSecond"


def test_clean_extracted_text_preserves_paragraph_breaks() -> None:
    assert clean_extracted_text("First paragraph.\n\nSecond paragraph.") == (
        "First paragraph.\n\nSecond paragraph."
    )


def test_clean_extracted_text_preserves_mathematical_expressions() -> None:
    expression = "∫ f(x) dx = α + β²; x ≤ 10"
    assert clean_extracted_text(expression) == expression


def test_clean_extracted_text_preserves_fenced_programming_code() -> None:
    code = '```python\nif  x ==  1:\n    print("two  spaces")\n```'
    assert clean_extracted_text(code) == code


def test_clean_extracted_text_preserves_multiple_blank_lines_in_fenced_code() -> None:
    code = "```text\nfirst\n\n\nsecond\n```"
    assert clean_extracted_text(code) == code


def test_clean_extracted_text_does_not_close_on_different_fence_marker() -> None:
    code = "~~~python\n  first   line\n````\n  second   line\n~~~"
    assert clean_extracted_text(code) == code


def test_clean_extracted_text_does_not_close_on_shorter_fence() -> None:
    code = "~~~~python\n  first   line\n~~~\n  second   line\n~~~~"
    assert clean_extracted_text(code) == code


def test_clean_extracted_text_accepts_closing_fence_longer_than_opening() -> None:
    code = "```python\n  first   line\n  second   line\n````"
    assert clean_extracted_text(code) == code


def test_clean_extracted_text_treats_unclosed_fence_as_ordinary_text() -> None:
    text = "```python\nlet   value =  1;\nmore   ordinary text"
    assert clean_extracted_text(text) == (
        "```python\nlet value = 1;\nmore ordinary text"
    )


def test_clean_extracted_text_preserves_unicode_characters() -> None:
    text = "café — 東京 — Ελληνικά — 🙂"
    assert clean_extracted_text(text) == text


def test_clean_extracted_text_returns_empty_for_whitespace_only() -> None:
    assert clean_extracted_text(" \t\n \t ") == ""


def test_clean_extracted_text_leaves_already_clean_text_unchanged() -> None:
    text = "A clean line.\nAnother clean line."
    assert clean_extracted_text(text) == text


def test_extract_pdf_text_cleans_each_page_and_preserves_numbers_and_status() -> None:
    class Page:
        def __init__(self, text: str | None) -> None:
            self.text = text

        def extract_text(self) -> str | None:
            return self.text

    class Reader:
        pages = [Page("  First   page  "), Page(" \t "), Page("Second\n\n\nparagraph")]

    result = extract_pdf_text(Reader())  # type: ignore[arg-type]

    assert result == {
        "page_count": 3,
        "text_status": "partial_text",
        "pages": [
            {"page_number": 1, "text": "First page"},
            {"page_number": 2, "text": ""},
            {"page_number": 3, "text": "Second\n\nparagraph"},
        ],
    }


def test_extract_pdf_text_reports_no_text_for_whitespace_page() -> None:
    class Page:
        def extract_text(self) -> str:
            return " \t\n "

    class Reader:
        pages = [Page()]

    result = extract_pdf_text(Reader())  # type: ignore[arg-type]

    assert result["text_status"] == "no_text_found"
    assert result["pages"] == [{"page_number": 1, "text": ""}]
