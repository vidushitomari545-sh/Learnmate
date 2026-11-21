export function formatUploadFailure(status, detail) {
  const message = typeof detail === "string" ? detail : "";
  const normalizedDetail = message.toLowerCase();

  if (status >= 500) {
    return `The server could not process the upload (HTTP ${status}).`;
  }

  let summary = "The upload was rejected.";
  if (status === 400) {
    summary = "File is empty.";
  } else if (status === 413) {
    summary = "File exceeds the upload limit.";
  } else if (status === 415) {
    summary =
      normalizedDetail.includes("signature") ||
      normalizedDetail.includes("valid pdf")
        ? "Invalid PDF content."
        : "Unsupported file type.";
  } else if (status === 422) {
    summary = "Choose a file to upload.";
  }

  return message
    ? `${summary} (HTTP ${status}: ${message})`
    : `${summary} (HTTP ${status}).`;
}

export function formatUnexpectedUploadResponse(status, reason) {
  return `Upload succeeded with HTTP ${status}, but the server response ${reason}. Check that the frontend and backend versions match.`;
}

export function isExtractedPdfResponse(result) {
  return (
    result !== null &&
    typeof result === "object" &&
    result.status === "extracted" &&
    typeof result.filename === "string" &&
    Number.isSafeInteger(result.size_bytes) &&
    result.size_bytes >= 0 &&
    Number.isSafeInteger(result.page_count) &&
    result.page_count >= 0 &&
    ["text_found", "partial_text", "no_text_found"].includes(
      result.text_status,
    ) &&
    Array.isArray(result.pages) &&
    result.pages.length === result.page_count &&
    result.pages.every(
      (page, index) =>
        page !== null &&
        typeof page === "object" &&
        page.page_number === index + 1 &&
        typeof page.text === "string",
    )
  );
}

export function isStoredUploadResponse(result) {
  return (
    result !== null &&
    typeof result === "object" &&
    result.status === "stored" &&
    typeof result.id === "string" &&
    typeof result.filename === "string" &&
    typeof result.uploaded_at === "string" &&
    Number.isFinite(Date.parse(result.uploaded_at)) &&
    Number.isSafeInteger(result.size_bytes) &&
    result.size_bytes >= 0 &&
    Number.isSafeInteger(result.page_count) &&
    result.page_count >= 0 &&
    ["text_found", "partial_text", "no_text_found"].includes(
      result.text_status,
    )
  );
}

export function mergeDocument(documents, incoming) {
  return [incoming, ...documents.filter((document) => document.id !== incoming.id)];
}

export function mergeDocumentPage(documents, incomingDocuments) {
  const knownIds = new Set(documents.map((document) => document.id));
  const uniqueIncoming = incomingDocuments.filter((document) => {
    if (knownIds.has(document.id)) {
      return false;
    }
    knownIds.add(document.id);
    return true;
  });
  return [...documents, ...uniqueIncoming];
}

export function getDocumentLoadState({ append, cursor = null, error = null }) {
  if (!error) {
    return {
      documentsState: "loaded",
      documentsError: "",
      paginationError: "",
      failedCursor: null,
    };
  }

  const message =
    error instanceof Error ? error.message : "Could not load study materials.";
  return append
    ? {
        documentsState: "loaded",
        documentsError: "",
        paginationError: message,
        failedCursor: cursor,
      }
    : {
        documentsState: "error",
        documentsError: message,
        paginationError: "",
        failedCursor: null,
      };
}

export function getDocumentPageRetryCursor(failedCursor, nextCursor) {
  return failedCursor ?? nextCursor;
}

export function getDocumentCountAfterUpload(count, documents, incoming) {
  return count + Number(!documents.some((document) => document.id === incoming.id));
}

export function getIdempotencyKey(currentKey, createKey = () => crypto.randomUUID()) {
  return currentKey || createKey();
}

export function getUploadConnectionError(error) {
  if (error instanceof TypeError) {
    return "Server connection failed. Check that the backend is running and try again.";
  }

  return error instanceof Error ? error.message : "Unable to upload this PDF.";
}
