import assert from "node:assert/strict";
import test from "node:test";

import {
  requestAnswerAndClearDraft,
  submitChatFormOnEnter,
} from "./chatSubmission.js";

function keyboardEvent({ key = "Enter", shiftKey = false, isComposing = false } = {}) {
  let preventDefaultCount = 0;
  let requestSubmitCount = 0;

  const event = {
    key,
    shiftKey,
    nativeEvent: { isComposing },
    preventDefault() {
      preventDefaultCount += 1;
    },
    currentTarget: {
      form: {
        requestSubmit() {
          requestSubmitCount += 1;
        },
      },
    },
  };

  return {
    event,
    get preventDefaultCount() {
      return preventDefaultCount;
    },
    get requestSubmitCount() {
      return requestSubmitCount;
    },
  };
}

test("Enter submits the existing chat form", () => {
  const keyboard = keyboardEvent();

  submitChatFormOnEnter(keyboard.event);

  assert.equal(keyboard.preventDefaultCount, 1);
  assert.equal(keyboard.requestSubmitCount, 1);
});

test("Shift+Enter does not submit or prevent the newline", () => {
  const keyboard = keyboardEvent({ shiftKey: true });

  submitChatFormOnEnter(keyboard.event);

  assert.equal(keyboard.preventDefaultCount, 0);
  assert.equal(keyboard.requestSubmitCount, 0);
});

test("Enter during IME composition does not submit or prevent the key", () => {
  const keyboard = keyboardEvent({ isComposing: true });

  submitChatFormOnEnter(keyboard.event);

  assert.equal(keyboard.preventDefaultCount, 0);
  assert.equal(keyboard.requestSubmitCount, 0);
});

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
