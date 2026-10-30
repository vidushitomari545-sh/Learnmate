import assert from "node:assert/strict";
import test from "node:test";

import {
  formatUnexpectedUploadResponse,
  formatUploadFailure,
  getUploadConnectionError,
  isExtractedPdfResponse,
} from "./uploadErrors.js";

test("formats unsupported file type with the backend detail and status", () => {
  assert.equal(
    formatUploadFailure(415, "Only PDF files are accepted."),
    "Unsupported file type. (HTTP 415: Only PDF files are accepted.)",
  );
});

test("formats invalid PDF content with the backend detail and status", () => {
  assert.equal(
    formatUploadFailure(415, "The uploaded file is not a valid PDF."),
    "Invalid PDF content. (HTTP 415: The uploaded file is not a valid PDF.)",
  );
});

test("formats empty and oversized upload errors", () => {
  assert.equal(
    formatUploadFailure(400, "The uploaded file is empty."),
    "File is empty. (HTTP 400: The uploaded file is empty.)",
  );
  assert.equal(
    formatUploadFailure(413, "PDF files must be 10 MB or smaller."),
    "File exceeds the upload limit. (HTTP 413: PDF files must be 10 MB or smaller.)",
  );
});

test("does not display internal server error details", () => {
  assert.equal(
    formatUploadFailure(500, "Traceback: secret internal details"),
    "The server could not process the upload (HTTP 500).",
  );
});

test("identifies a failed connection", () => {
  assert.equal(
    getUploadConnectionError(new TypeError("Failed to fetch")),
    "Server connection failed. Check that the backend is running and try again.",
  );
});

test("validates the extracted PDF response structure", () => {
  const phase23Response = {
    id: "document-id",
    filename: "notes.pdf",
    status: "stored",
    size_bytes: 123,
  };
  assert.equal(
    isExtractedPdfResponse(phase23Response),
    false,
  );
  assert.equal(
    isExtractedPdfResponse({
      filename: "notes.pdf",
      size_bytes: 123,
      status: "extracted",
      page_count: 2,
      text_status: "partial_text",
      pages: [
        { page_number: 1, text: "Readable text" },
        { page_number: 2, text: "" },
      ],
    }),
    true,
  );
});
