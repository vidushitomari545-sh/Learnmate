import assert from "node:assert/strict";
import test from "node:test";

import { requestAnswerAndClearDraft } from "./chatSubmission.js";

test("clears the draft after a successful chat response", async () => {
  const answer = { answer: "A clear answer." };
  let draft = "What is this?";

  const result = await requestAnswerAndClearDraft(
    async () => answer,
    () => {
      draft = "";
    },
  );

  assert.equal(result, answer);
  assert.equal(draft, "");
});

test("preserves the draft when the chat request fails", async () => {
  const error = new Error("Chat request failed.");
  let draft = "What is this?";
  let clearCount = 0;

  await assert.rejects(
    requestAnswerAndClearDraft(
      async () => {
        throw error;
      },
      () => {
        clearCount += 1;
        draft = "";
      },
    ),
    error,
  );

  assert.equal(draft, "What is this?");
  assert.equal(clearCount, 0);
});
