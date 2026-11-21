const httpErrorMessages = {
  422: "The question was invalid. Please check it and try again.",
  429: "The chat service is temporarily rate limited. Please try again later.",
  502: "The chat service returned an invalid response. Please try again.",
  503: "The chat service is temporarily unavailable. Please try again.",
  504: "The chat service timed out. Please try again.",
};

function isAbortError(error) {
  return error !== null && typeof error === "object" && error.name === "AbortError";
}

function normalizeChatResponse(payload) {
  if (
    payload === null ||
    typeof payload !== "object" ||
    typeof payload.answer !== "string" ||
    typeof payload.insufficient_evidence !== "boolean" ||
    !Array.isArray(payload.citations) ||
    payload.citations.some(
      (citation) =>
        citation === null ||
        typeof citation !== "object" ||
        typeof citation.document_name !== "string" ||
        !Array.isArray(citation.page_numbers) ||
        !citation.page_numbers.every(
          (pageNumber) => Number.isSafeInteger(pageNumber) && pageNumber > 0,
        ),
    )
  ) {
    throw new Error("The chat service returned an invalid response.");
  }

  return {
    answer: payload.answer,
    insufficientEvidence: payload.insufficient_evidence,
    citations: payload.citations.map((citation) => ({
      documentName: citation.document_name,
      pageNumbers: [...citation.page_numbers],
    })),
  };
}

export async function requestChatAnswer(question, { signal } = {}) {
  if (typeof question !== "string" || !question.trim()) {
    throw new Error("Enter a question before sending it.");
  }

  let response;
  try {
    response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question.trim() }),
      signal,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw error;
    }
    if (error instanceof TypeError) {
      throw new Error(
        "Could not connect to the chat service. Check your connection and try again.",
      );
    }
    throw error;
  }

  if (!response.ok) {
    throw new Error(
      httpErrorMessages[response.status] ??
        `The chat request failed (HTTP ${response.status}).`,
    );
  }

  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new Error("The chat service returned malformed JSON.");
  }

  return normalizeChatResponse(payload);
}
