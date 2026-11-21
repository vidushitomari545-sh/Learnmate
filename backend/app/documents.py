from dataclasses import dataclass
import re
from typing import BinaryIO, Literal, TypedDict

from pypdf import PdfReader
from pypdf.errors import PdfReadError

MAX_DOCUMENT_SIZE = 10 * 1024 * 1024
CHUNK_SIZE = 64 * 1024


class InvalidPDFError(ValueError):
    """Raised when an uploaded file is not a readable PDF."""


class EmptyFileError(ValueError):
    """Raised when an uploaded file has no content."""


class DocumentTooLargeError(ValueError):
    """Raised when an uploaded PDF exceeds the configured size limit."""


class PDFTextExtractionError(ValueError):
    """Raised when text cannot be safely extracted from a PDF."""


class ExtractedPage(TypedDict):
    page_number: int
    text: str


class PDFExtractionResult(TypedDict):
    page_count: int
    text_status: Literal["text_found", "partial_text", "no_text_found"]
    pages: list[ExtractedPage]


@dataclass
class ValidatedPDF:
    size_bytes: int
    reader: PdfReader


def clean_extracted_text(text: str | None) -> str:
    """Clean extracted text while preserving whitespace in properly fenced code.

    An unmatched fence is treated as ordinary text; it does not mark the
    remainder of the page as code.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", " ").replace("\x0c", "\n")

    lines = text.split("\n")
    cleaned_lines: list[tuple[str, bool]] = []
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        stripped_line = line.strip()
        fence_match = re.match(r"(`{3,}|~{3,})", stripped_line)
        if fence_match is not None:
            opening_fence = fence_match.group(1)
            marker = opening_fence[0]
            closing_pattern = re.compile(r"(`+|~+)\s*\Z")
            closing_index = next(
                (
                    candidate_index
                    for candidate_index in range(line_index + 1, len(lines))
                    if (
                        (closing_match := closing_pattern.fullmatch(
                            lines[candidate_index].strip()
                        ))
                        is not None
                        and closing_match.group(1)[0] == marker
                        and len(closing_match.group(1)) >= len(opening_fence)
                    )
                ),
                None,
            )
            if closing_index is not None:
                cleaned_lines.append((stripped_line, True))
                cleaned_lines.extend(
                    (code_line.rstrip(), True)
                    for code_line in lines[line_index + 1 : closing_index]
                )
                cleaned_lines.append((lines[closing_index].strip(), True))
                line_index = closing_index + 1
                continue

        indentation = line[: len(line) - len(line.lstrip(" \t"))]
        is_indented_code = len(indentation.expandtabs(4)) >= 4
        if is_indented_code:
            cleaned_lines.append((line.rstrip(), False))
        else:
            cleaned_lines.append((re.sub(r"[ \t]+", " ", line).strip(), False))
        line_index += 1

    collapsed_lines: list[str] = []
    previous_outside_blank = False
    for line, is_fenced_code in cleaned_lines:
        is_outside_blank = not is_fenced_code and not line
        if is_outside_blank and previous_outside_blank:
            continue
        collapsed_lines.append(line)
        previous_outside_blank = is_outside_blank

    cleaned_text = "\n".join(collapsed_lines)
    return cleaned_text.strip("\n")


def validate_pdf_upload(
    filename: str | None, content_type: str | None, file: BinaryIO
) -> ValidatedPDF:
    if filename is None or not filename.lower().endswith(".pdf"):
        raise InvalidPDFError("Only PDF files are accepted.")
    if (
        content_type is None
        or content_type.split(";", 1)[0].strip().lower() != "application/pdf"
    ):
        raise InvalidPDFError("The content type must be application/pdf.")

    file.seek(0)
    signature = file.read(5)
    if not signature:
        raise EmptyFileError("The uploaded file is empty.")

    size_bytes = len(signature)
    while chunk := file.read(min(CHUNK_SIZE, MAX_DOCUMENT_SIZE - size_bytes + 1)):
        size_bytes += len(chunk)
        if size_bytes > MAX_DOCUMENT_SIZE:
            raise DocumentTooLargeError(
                "PDF files must be "
                f"{MAX_DOCUMENT_SIZE // (1024 * 1024)} MB or smaller."
            )

    if signature != b"%PDF-":
        raise InvalidPDFError("The uploaded file does not contain a PDF signature.")

    file.seek(0)
    try:
        reader = PdfReader(file, strict=True)
        if reader.is_encrypted or len(reader.pages) == 0:
            raise InvalidPDFError("The uploaded file is not a valid PDF.")
    except (
        PdfReadError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        UnicodeError,
        EOFError,
        OSError,
    ) as error:
        raise InvalidPDFError("The uploaded file is not a valid PDF.") from error
    finally:
        file.seek(0)

    return ValidatedPDF(size_bytes=size_bytes, reader=reader)


def extract_pdf_text(reader: PdfReader) -> PDFExtractionResult:
    pages: list[ExtractedPage] = []
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            pages.append(
                {
                    "page_number": page_number,
                    "text": clean_extracted_text(page.extract_text()),
                }
            )
    except (
        PdfReadError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        UnicodeError,
        EOFError,
        OSError,
        NotImplementedError,
    ) as error:
        raise PDFTextExtractionError(
            "The PDF could not be read for text extraction."
        ) from error

    pages_with_text = sum(bool(page["text"].strip()) for page in pages)
    if pages_with_text == 0:
        text_status: Literal[
            "text_found", "partial_text", "no_text_found"
        ] = "no_text_found"
    elif pages_with_text < len(pages):
        text_status = "partial_text"
    else:
        text_status = "text_found"

    return {
        "page_count": len(pages),
        "text_status": text_status,
        "pages": pages,
    }
