const jobStatuses = new Set(["queued", "processing", "completed", "failed"]);
const jobPhases = new Set([
  "queued",
  "mapping",
  "reducing",
  "finalizing",
  "completed",
  "failed",
]);
const textStatuses = new Set(["text_found", "partial_text", "no_text_found"]);

function isPositivePageList(value) {
  return (
    Array.isArray(value) &&
    value.every((page) => Number.isSafeInteger(page) && page > 0) &&
    value.every((page, index) => index === 0 || value[index - 1] < page)
  );
}

function normalizeSummaryResult(result, pageCount) {
  if (
    result === null ||
    typeof result !== "object" ||
    typeof result.summary_markdown !== "string" ||
    !result.summary_markdown.trim() ||
    !isPositivePageList(result.page_references) ||
    !isPositivePageList(result.pages_without_text) ||
    !Number.isSafeInteger(result.chunk_count) ||
    result.chunk_count < 1 ||
    !["text_found", "partial_text"].includes(result.text_status)
  ) {
    throw new Error("The summary service returned an invalid result.");
  }

  const coveredPages = [
    ...result.page_references,
    ...result.pages_without_text,
  ].sort((left, right) => left - right);
  if (
    coveredPages.length !== pageCount ||
    coveredPages.some((page, index) => page !== index + 1) ||
    result.text_status !==
      (result.pages_without_text.length > 0 ? "partial_text" : "text_found")
  ) {
    throw new Error("The summary service returned invalid page coverage.");
  }

  return {
    summaryMarkdown: result.summary_markdown,
    pageReferences: [...result.page_references],
    pagesWithoutText: [...result.pages_without_text],
    chunkCount: result.chunk_count,
    textStatus: result.text_status,
  };
}

function normalizeSummaryJob(payload, expectedDocumentId) {
  if (
    payload === null ||
    typeof payload !== "object" ||
    typeof payload.id !== "string" ||
    !payload.id ||
    payload.document_id !== expectedDocumentId ||
    typeof payload.filename !== "string" ||
    !Number.isSafeInteger(payload.page_count) ||
    payload.page_count < 0 ||
    !textStatuses.has(payload.text_status) ||
    !jobStatuses.has(payload.status) ||
    !jobPhases.has(payload.phase) ||
    !Number.isSafeInteger(payload.progress_current) ||
    !Number.isSafeInteger(payload.progress_total) ||
    payload.progress_current < 0 ||
    payload.progress_total < 0 ||
    payload.progress_current > payload.progress_total ||
    typeof payload.cache_hit !== "boolean" ||
    typeof payload.retryable !== "boolean" ||
    !(payload.error_code === null || typeof payload.error_code === "string")
  ) {
    throw new Error("The summary service returned an invalid job.");
  }

  const result =
    payload.status === "completed"
      ? normalizeSummaryResult(payload.result, payload.page_count)
      : null;
  if (payload.status !== "completed" && payload.result !== null) {
    throw new Error("The summary service returned an invalid job result.");
  }

  return {
    id: payload.id,
    documentId: payload.document_id,
    filename: payload.filename,
    pageCount: payload.page_count,
    textStatus: payload.text_status,
    status: payload.status,
    phase: payload.phase,
    progressCurrent: payload.progress_current,
    progressTotal: payload.progress_total,
    cacheHit: payload.cache_hit,
    retryable: payload.retryable,
    errorCode: payload.error_code,
    result,
  };
}

async function requestSummary(url, options, documentId) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    if (error?.name === "AbortError") {
      throw error;
    }
    if (error instanceof TypeError) {
      throw new Error(
        "Could not connect to the document summary service. Check your connection and try again.",
      );
    }
    throw error;
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The summary service returned malformed JSON.");
  }
  if (!response.ok) {
    const detail =
      payload &&
      typeof payload === "object" &&
      typeof payload.detail === "string"
        ? payload.detail
        : `The summary request failed (HTTP ${response.status}).`;
    throw new Error(detail);
  }
  return normalizeSummaryJob(payload, documentId);
}

export function startDocumentSummary(documentId, { signal } = {}) {
  if (typeof documentId !== "string" || !documentId) {
    throw new Error("A document must be selected for summarization.");
  }
  return requestSummary(
    `/api/documents/${encodeURIComponent(documentId)}/summary`,
    { method: "POST", signal },
    documentId,
  );
}

export function getDocumentSummary(documentId, jobId, { signal } = {}) {
  if (typeof documentId !== "string" || !documentId) {
    throw new Error("A document must be selected for summarization.");
  }
  if (typeof jobId !== "string" || !jobId) {
    throw new Error("A summary job is required to check its status.");
  }
  return requestSummary(
    `/api/documents/${encodeURIComponent(documentId)}/summary/${encodeURIComponent(jobId)}`,
    { signal },
    documentId,
  );
}

export function formatPageReferences(pages) {
  if (!Array.isArray(pages) || pages.length === 0) {
    return "";
  }
  const ranges = [];
  let start = pages[0];
  let end = start;
  for (const page of pages.slice(1)) {
    if (page === end + 1) {
      end = page;
      continue;
    }
    ranges.push(start === end ? `${start}` : `${start}–${end}`);
    start = page;
    end = page;
  }
  ranges.push(start === end ? `${start}` : `${start}–${end}`);
  return ranges.join(", ");
}
