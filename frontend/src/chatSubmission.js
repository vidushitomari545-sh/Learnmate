export async function requestAnswerAndClearDraft(requestAnswer, clearDraft) {
  const answer = await requestAnswer();
  clearDraft();
  return answer;
}
