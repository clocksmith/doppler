// ============================================================================
// Chat Templates
// ============================================================================

// Simple prompt templates for single-turn chat.
// For multi-turn conversations, use formatChatMessages from chat-format.js.

function applyTurnBasedTemplate(prompt) {
  // Turn-based format: <start_of_turn>role\ncontent<end_of_turn>
  const userTurn = `<start_of_turn>user\n${prompt}<end_of_turn>\n`;
  const modelTurn = '<start_of_turn>model\n';
  return userTurn + modelTurn;
}

function applyHeaderBasedTemplate(prompt) {
  // Header-based format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
  return `<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n${prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n`;
}

function applyChannelBasedTemplate(prompt) {
  // Channel-based format: <|start|>role<|message|>content<|end|>
  return `<|start|>user<|message|>${prompt}<|end|><|start|>assistant<|channel|>final<|message|>`;
}

function applyChatMLTemplate(prompt) {
  // ChatML format: <|im_start|>role\ncontent<|im_end|>
  return `<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;
}

function getQwenAssistantPrefix(options) {
  return options?.thinking === true
    ? '<|im_start|>assistant\n<think>\n'
    : '<|im_start|>assistant\n<think>\n\n</think>\n\n';
}

function applyQwenTemplate(prompt, options) {
  return `<|im_start|>user\n${prompt}<|im_end|>\n${getQwenAssistantPrefix(options)}`;
}

function applyGemma4Template(prompt) {
  return `<bos><|turn>user\n${prompt}<turn|>\n<|turn>model\n`;
}

function applyTranslateGemmaTemplate() {
  throw new Error(
    'TranslateGemma template requires structured messages. ' +
    'Use formatChatMessages(messages, "translategemma") instead of applyChatTemplate(prompt, ...).'
  );
}

// Template type to formatter mapping.
// Add new template types here rather than adding switch cases.
const PROMPT_TEMPLATES = {
  'gemma': applyTurnBasedTemplate,
  'gemma4': applyGemma4Template,
  'llama3': applyHeaderBasedTemplate,
  'gpt-oss': applyChannelBasedTemplate,
  'chatml': applyChatMLTemplate,
  'qwen': applyQwenTemplate,
  'translategemma': applyTranslateGemmaTemplate,
};

export function applyChatTemplate(prompt, templateType, options = undefined) {
  if (templateType == null) {
    return prompt;
  }
  const formatter = PROMPT_TEMPLATES[templateType];
  if (formatter) {
    return formatter(prompt, options);
  }
  throw new Error(`Unrecognized chat template type: ${templateType}`);
}

// Exports preserved for existing external imports.
export const applyGemmaChatTemplate = applyTurnBasedTemplate;
export const applyGemma4ChatTemplate = applyGemma4Template;
export const applyLlama3ChatTemplate = applyHeaderBasedTemplate;
export const applyGptOssChatTemplate = applyChannelBasedTemplate;
export const applyQwenChatTemplate = applyQwenTemplate;


export function isStopToken(token, stopTokenIds, eosTokenId) {
  if (stopTokenIds.includes(token)) return true;
  if (typeof eosTokenId === 'number' && token === eosTokenId) return true;
  return false;
}
