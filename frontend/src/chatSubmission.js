export async function requestAnswerAndClearDraft(requestAnswer, clearDraft) {
  const answer = await requestAnswer();
  clearDraft();
  return answer;
}

export function submitChatFormOnEnter(event) {
  if (
    event.key !== "Enter" ||
    event.shiftKey ||
    event.nativeEvent.isComposing
  ) {
    return;
  }

  event.preventDefault();
  event.currentTarget.form?.requestSubmit();
}
