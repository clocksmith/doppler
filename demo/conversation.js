export const MAX_CONTEXT_HISTORY_TURNS = 16;

export function countConversationTurns(history) {
  if (!Array.isArray(history)) return 0;
  return history.reduce((count, message) => (
    message?.role === 'user' && typeof message.content === 'string'
      ? count + 1
      : count
  ), 0);
}

export function normalizeConversationHistory(history) {
  if (!Array.isArray(history)) return [];
  return history
    .filter((message) => (
      (message?.role === 'user' || message?.role === 'assistant')
      && typeof message.content === 'string'
    ))
    .map((message) => ({ role: message.role, content: message.content }));
}

function selectConversationContext(history) {
  return normalizeConversationHistory(history).slice(-(MAX_CONTEXT_HISTORY_TURNS * 2));
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
  const storedMessages = normalizeConversationHistory(history);
  const contextMessages = [
    ...selectConversationContext(storedMessages),
    { role: 'user', content: currentPrompt },
  ];
  const messages = [...storedMessages, { role: 'user', content: currentPrompt }];
  return {
    promptInput: {
      messages: toRuntimeMessages(contextMessages, options.templateType ?? null, options.translation),
    },
    messages,
    contextMessages,
    currentPrompt,
    priorTurnCount: countConversationTurns(contextMessages) - 1,
  };
}

export function appendConversationTurn(history, request, output) {
  const currentPrompt = typeof request.currentPrompt === 'string'
    ? request.currentPrompt.trim()
    : '';
  const assistantOutput = typeof output === 'string' ? output.trim() : '';
  if (!currentPrompt || !assistantOutput) {
    return normalizeConversationHistory(history);
  }
  return normalizeConversationHistory([
    ...(Array.isArray(history) ? history : []),
    { role: 'user', content: currentPrompt },
    { role: 'assistant', content: assistantOutput },
  ]);
}
