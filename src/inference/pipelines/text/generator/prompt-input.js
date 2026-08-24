import { formatChatMessages } from '../chat-format.js';
import { applyChatTemplate } from '../init.js';

function isStructuredChatRequest(prompt) {
  return prompt != null
    && typeof prompt === 'object'
    && !Array.isArray(prompt)
    && Array.isArray(prompt.messages);
}

export function resolvePromptInput(state, prompt, useChatTemplate, contextLabel) {
  const chatOptions = state.modelConfig.chatTemplateThinking === true ? { thinking: true } : undefined;
  if (typeof prompt === 'string') {
    if (useChatTemplate && state.modelConfig.chatTemplateType) {
      if (state.modelConfig.chatTemplateType === 'translategemma') {
        throw new Error(
          `[Pipeline] ${contextLabel}: translategemma chat template requires structured messages. ` +
          'Pass { messages: [...] } instead of a plain string prompt.'
        );
      }
      return applyChatTemplate(prompt, state.modelConfig.chatTemplateType, chatOptions);
    }
    return prompt;
  }
  if (
    prompt != null
    && typeof prompt === 'object'
    && !Array.isArray(prompt)
    && 'messages' in prompt
    && !Array.isArray(prompt.messages)
  ) {
    throw new Error(
      `[Pipeline] ${contextLabel}: prompt.messages must be an array of chat messages, got ${typeof prompt.messages}. ` +
      'Pass { messages: [{ role: "user", content: "..." }, ...] }.'
    );
  }
  const messages = isStructuredChatRequest(prompt)
    ? prompt.messages
    : (Array.isArray(prompt) ? prompt : null);
  if (!messages) {
    throw new Error(
      `[Pipeline] ${contextLabel}: prompt must be a string, chat message array, or { messages: [...] }.`
    );
  }
  const templateType = useChatTemplate ? state.modelConfig.chatTemplateType : null;
  return formatChatMessages(messages, templateType, chatOptions);
}
