import assert from "node:assert/strict";
import test from "node:test";

import {
  formatUnexpectedUploadResponse,
  formatUploadFailure,
  getDocumentCountAfterUpload,
  getDocumentLoadState,
  getDocumentPageRetryCursor,
  getIdempotencyKey,
  getUploadConnectionError,
  isExtractedPdfResponse,
  isStoredUploadResponse,
  mergeDocument,
  mergeDocumentPage,
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

test("validates a stored upload response", () => {
  assert.equal(
    isStoredUploadResponse({
      id: "document-id",
      filename: "notes.pdf",
      uploaded_at: "2026-10-31T10:00:00+00:00",
      size_bytes: 123,
      status: "stored",
      page_count: 2,
      text_status: "partial_text",
    }),
    true,
  );
  assert.equal(
    isStoredUploadResponse({
      filename: "notes.pdf",
      size_bytes: 123,
      status: "extracted",
    }),
    false,
  );
});

test("merges a successful upload without duplicating its document id", () => {
  const existing = { id: "old", filename: "old.pdf" };
  const previousUpload = { id: "new", filename: "old-name.pdf" };
  const retryResult = { id: "new", filename: "renamed.pdf" };

  assert.deepEqual(
    mergeDocument([existing, previousUpload], retryResult),
    [retryResult, existing],
  );
});

test("appends a paginated document page without reversing or duplicating it", () => {
  const current = [{ id: "newest" }, { id: "middle" }];
  const page = [{ id: "older" }, { id: "middle" }];

  assert.deepEqual(
    mergeDocumentPage(current, page),
    [{ id: "newest" }, { id: "middle" }, { id: "older" }],
  );
});

test("keeps the first page and retries a failed page with its original cursor", async () => {
  const firstPage = {
    documents: [{ id: "first" }, { id: "shared" }],
    next_cursor: "page-two-cursor",
    total_count: 4,
  };
  const secondPage = {
    documents: [{ id: "second" }, { id: "shared" }],
    next_cursor: null,
    total_count: 4,
  };
  const requestedCursors = [];
  let pageAttempts = 0;
  const fetchPage = async (cursor) => {
    requestedCursors.push(cursor);
    pageAttempts += 1;
    if (pageAttempts === 1) {
      return firstPage;
    }
    if (pageAttempts === 2) {
      throw new Error("Could not load study materials (503).");
    }
    return secondPage;
  };

  const loadedFirstPage = await fetchPage(null);
  let documents = loadedFirstPage.documents;
  let paginationState = getDocumentLoadState({ append: false });

  try {
    await fetchPage(loadedFirstPage.next_cursor);
  } catch (error) {
    paginationState = getDocumentLoadState({
      append: true,
      cursor: loadedFirstPage.next_cursor,
      error,
    });
  }

  assert.equal(paginationState.documentsState, "loaded");
  assert.equal(paginationState.paginationError, "Could not load study materials (503).");
  assert.deepEqual(documents, firstPage.documents);

  const retryCursor = getDocumentPageRetryCursor(
    paginationState.failedCursor,
    loadedFirstPage.next_cursor,
  );
  const retriedPage = await fetchPage(retryCursor);
  documents = mergeDocumentPage(documents, retriedPage.documents);
  paginationState = getDocumentLoadState({ append: true });

  assert.deepEqual(requestedCursors, [null, "page-two-cursor", "page-two-cursor"]);
  assert.deepEqual(documents, [
    { id: "first" },
    { id: "shared" },
    { id: "second" },
  ]);
  assert.equal(paginationState.paginationError, "");
  assert.equal(paginationState.failedCursor, null);
});

test("initial document-list failures keep the existing full-list error state", () => {
  assert.deepEqual(
    getDocumentLoadState({
      append: false,
      error: new Error("Could not load study materials (500)."),
    }),
    {
      documentsState: "error",
      documentsError: "Could not load study materials (500).",
      paginationError: "",
      failedCursor: null,
    },
  );
});

test("increments the library count only for a newly stored document", () => {
  const documents = [{ id: "saved" }];

  assert.equal(
    getDocumentCountAfterUpload(1, documents, { id: "new" }),
    2,
  );
  assert.equal(
    getDocumentCountAfterUpload(1, documents, { id: "saved" }),
    1,
  );
});

test("reuses the same idempotency key for upload retries", () => {
  let created = 0;
  const firstKey = getIdempotencyKey(null, () => {
    created += 1;
    return "retry-key";
  });
  const retryKey = getIdempotencyKey(firstKey, () => {
    created += 1;
    return "different-key";
  });

  assert.equal(firstKey, "retry-key");
  assert.equal(retryKey, firstKey);
  assert.equal(created, 1);
});
