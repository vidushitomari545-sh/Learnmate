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

export function getUploadConnectionError(error) {
  if (error instanceof TypeError) {
    return "Server connection failed. Check that the backend is running and try again.";
  }

  return error instanceof Error ? error.message : "Unable to upload this PDF.";
}
