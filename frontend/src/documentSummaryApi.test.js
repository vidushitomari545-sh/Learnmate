import assert from "node:assert/strict";
import test from "node:test";

import {
  formatPageReferences,
  getDocumentSummary,
  startDocumentSummary,
} from "./documentSummaryApi.js";

function mockFetch(t, implementation) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = implementation;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
}

function response(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  };
}

function summaryJob(overrides = {}) {
  return {
    id: "job-1",
    document_id: "doc-1",
    filename: "notes.pdf",
    page_count: 3,
    text_status: "partial_text",
    status: "completed",
    phase: "completed",
    progress_current: 2,
    progress_total: 2,
    cache_hit: false,
    retryable: false,
    error_code: null,
    result: {
      summary_markdown: "## Topic-wise explanations\n\n## Final revision\n",
      page_references: [1, 3],
      pages_without_text: [2],
      chunk_count: 2,
      text_status: "partial_text",
    },
    ...overrides,
  };
}

test("starts and normalizes a completed full-document summary", async (t) => {
  let request;
  mockFetch(t, async (url, options) => {
    request = { url, options };
    return response(summaryJob());
  });

  const result = await startDocumentSummary("doc-1");

  assert.equal(request.url, "/api/documents/doc-1/summary");
  assert.equal(request.options.method, "POST");
  assert.equal(result.documentId, "doc-1");
  assert.equal(result.status, "completed");
  assert.deepEqual(result.result.pageReferences, [1, 3]);
  assert.deepEqual(result.result.pagesWithoutText, [2]);
});

test("polls a job and safely encodes document and job IDs", async (t) => {
  let request;
  mockFetch(t, async (url, options) => {
    request = { url, options };
    return response(
      summaryJob({
        document_id: "doc / 1",
        id: "job / 2",
        status: "processing",
        phase: "mapping",
        progress_current: 1,
        progress_total: 2,
        result: null,
      }),
    );
  });

  const controller = new AbortController();
  const result = await getDocumentSummary("doc / 1", "job / 2", {
    signal: controller.signal,
  });

  assert.equal(
    request.url,
    "/api/documents/doc%20%2F%201/summary/job%20%2F%202",
  );
  assert.equal(request.options.signal, controller.signal);
  assert.equal(result.status, "processing");
  assert.equal(result.result, null);
});

test("rejects incomplete or duplicate page provenance", async (t) => {
  let payload = summaryJob({
    result: { ...summaryJob().result, page_references: [1, 1] },
  });
  mockFetch(t, async () => response(payload));

  await assert.rejects(startDocumentSummary("doc-1"), /invalid result/);

  payload = summaryJob({
    result: {
      ...summaryJob().result,
      page_references: [1],
      pages_without_text: [],
      text_status: "text_found",
    },
  });
  await assert.rejects(startDocumentSummary("doc-1"), /page coverage/);
});

test("rejects a response for a different document", async (t) => {
  mockFetch(t, async () => response(summaryJob({ document_id: "other-doc" })));
  await assert.rejects(startDocumentSummary("doc-1"), /invalid job/);
});

test("reports API errors and malformed JSON", async (t) => {
  let fetchResponse = response({ detail: "Document not found." }, 404);
  mockFetch(t, async () => fetchResponse);
  await assert.rejects(startDocumentSummary("doc-1"), /Document not found/);

  fetchResponse = {
    ok: true,
    status: 200,
    json: async () => {
      throw new SyntaxError("Unexpected token");
    },
  };
  await assert.rejects(startDocumentSummary("doc-1"), /malformed JSON/);
});

test("formats page references as compact ranges", () => {
  assert.equal(formatPageReferences([1, 2, 3, 5, 7, 8]), "1–3, 5, 7–8");
  assert.equal(formatPageReferences([]), "");
});
