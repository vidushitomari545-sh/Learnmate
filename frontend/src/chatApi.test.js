import assert from "node:assert/strict";
import test from "node:test";

import { requestChatAnswer } from "./chatApi.js";

function mockFetch(t, implementation) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = implementation;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });
}

function jsonResponse(payload, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => payload,
  };
}

test("sends a question and normalizes a valid response", async (t) => {
  let request;
  mockFetch(t, async (url, options) => {
    request = { url, options };
    return jsonResponse({
      answer: "A clear answer.",
      insufficient_evidence: false,
      citations: [{ document_name: "Notes.pdf", page_numbers: [1, 2] }],
    });
  });

  assert.deepEqual(await requestChatAnswer("  What is this?  "), {
    answer: "A clear answer.",
    insufficientEvidence: false,
    citations: [{ documentName: "Notes.pdf", pageNumbers: [1, 2] }],
  });
  assert.equal(request.url, "/api/chat");
  assert.equal(request.options.method, "POST");
  assert.equal(request.options.headers["Content-Type"], "application/json");
  assert.deepEqual(JSON.parse(request.options.body), { question: "What is this?" });
});

test("normalizes an insufficient-evidence response", async (t) => {
  mockFetch(t, async () =>
    jsonResponse({
      answer: "I could not find this in your materials.",
      insufficient_evidence: true,
      citations: [],
    }),
  );

  assert.deepEqual(await requestChatAnswer("What is not in my notes?"), {
    answer: "I could not find this in your materials.",
    insufficientEvidence: true,
    citations: [],
  });
});

test("rejects an invalid response schema", async (t) => {
  mockFetch(t, async () =>
    jsonResponse({
      answer: "An answer",
      insufficient_evidence: false,
      citations: [{ document_name: "Notes.pdf", page_numbers: ["1"] }],
    }),
  );

  await assert.rejects(
    requestChatAnswer("A question"),
    /invalid response/,
  );
});

for (const [status, message] of [
  [422, /question was invalid/],
  [429, /rate limited/],
  [502, /invalid response/],
  [503, /temporarily unavailable/],
  [504, /timed out/],
]) {
  test(`reports HTTP ${status} errors`, async (t) => {
    mockFetch(t, async () => jsonResponse({}, status));
    await assert.rejects(requestChatAnswer("A question"), message);
  });
}

test("reports malformed JSON", async (t) => {
  mockFetch(t, async () => ({
    ok: true,
    status: 200,
    json: async () => {
      throw new SyntaxError("Unexpected token");
    },
  }));

  await assert.rejects(requestChatAnswer("A question"), /malformed JSON/);
});

test("reports network failures", async (t) => {
  mockFetch(t, async () => {
    throw new TypeError("Failed to fetch");
  });

  await assert.rejects(
    requestChatAnswer("A question"),
    /Could not connect to the chat service/,
  );
});

test("passes through request cancellation", async (t) => {
  const controller = new AbortController();
  const abortError = new DOMException("The request was aborted.", "AbortError");
  mockFetch(t, async (_url, options) => {
    assert.equal(options.signal, controller.signal);
    controller.abort();
    throw abortError;
  });

  await assert.rejects(requestChatAnswer("A question", { signal: controller.signal }), {
    name: "AbortError",
  });
});
