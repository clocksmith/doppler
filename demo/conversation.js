export const DEFAULT_HISTORY_TURN_LIMIT = 8;
export const MAX_STORED_HISTORY_TURNS = 16;

export function normalizeHistoryTurnLimit(value, fallback = DEFAULT_HISTORY_TURN_LIMIT) {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0
    ? Math.min(parsed, MAX_STORED_HISTORY_TURNS)
    : fallback;
}

export function countConversationTurns(history) {
  if (!Array.isArray(history)) return 0;
  return history.reduce((count, message) => (
    message?.role === 'user' && typeof message.content === 'string'
      ? count + 1
      : count
  ), 0);
}

export function trimConversationHistory(history, turnLimit = MAX_STORED_HISTORY_TURNS) {
  if (!Array.isArray(history)) return [];
  const maxMessages = normalizeHistoryTurnLimit(turnLimit, MAX_STORED_HISTORY_TURNS) * 2;
  return history
    .filter((message) => (
      (message?.role === 'user' || message?.role === 'assistant')
      && typeof message.content === 'string'
    ))
    .slice(-maxMessages)
    .map((message) => ({ role: message.role, content: message.content }));
}

function toRuntimeMessages(messages, templateType, translation) {
  if (templateType !== 'translategemma') {
    return messages.map((message) => ({ ...message }));
  }
  const sourceLangCode = translation?.sourceLangCode ?? 'en';
  const targetLangCode = translation?.targetLangCode ?? 'es';
  return messages.map((message) => {
    if (message.role !== 'user') return { ...message };
    return {
      role: 'user',
      content: [{
        type: 'text',
        text: message.content,
        source_lang_code: sourceLangCode,
        target_lang_code: targetLangCode,
      }],
    };
  });
}

export function createConversationRequest(history, prompt, options = {}) {
  const currentPrompt = typeof prompt === 'string' ? prompt.trim() : '';
  if (!currentPrompt) {
    throw new Error('Conversation prompt is required.');
  }
  const historyEnabled = options.historyEnabled !== false;
  const turnLimit = normalizeHistoryTurnLimit(options.turnLimit);
  const previousMessages = historyEnabled
    ? trimConversationHistory(history, turnLimit)
    : [];
  const messages = [
    ...previousMessages,
    { role: 'user', content: currentPrompt },
  ];
  return {
    promptInput: {
      messages: toRuntimeMessages(messages, options.templateType ?? null, options.translation),
    },
    messages,
    currentPrompt,
    historyEnabled,
    turnLimit,
    priorTurnCount: countConversationTurns(previousMessages),
  };
}

export function appendConversationTurn(history, request, output) {
  if (request?.historyEnabled !== true) {
    return trimConversationHistory(history, MAX_STORED_HISTORY_TURNS);
  }
  const currentPrompt = typeof request.currentPrompt === 'string'
    ? request.currentPrompt.trim()
    : '';
  const assistantOutput = typeof output === 'string' ? output.trim() : '';
  if (!currentPrompt || !assistantOutput) {
    return trimConversationHistory(history, MAX_STORED_HISTORY_TURNS);
  }
  return trimConversationHistory([
    ...(Array.isArray(history) ? history : []),
    { role: 'user', content: currentPrompt },
    { role: 'assistant', content: assistantOutput },
  ], MAX_STORED_HISTORY_TURNS);
}
