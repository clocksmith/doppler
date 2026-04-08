// Chat template formatters keyed by template type.
// Template types are stored in manifest.inference.chatTemplate.type.

const TRANSLATEGEMMA_LANGUAGE_ENTRIES = Object.freeze([
  ['ar-EG', 'Arabic'],
  ['ar-SA', 'Arabic'],
  ['bg-BG', 'Bulgarian'],
  ['bn-IN', 'Bengali'],
  ['ca-ES', 'Catalan'],
  ['cs-CZ', 'Czech'],
  ['da-DK', 'Danish'],
  ['de-DE', 'German'],
  ['el-GR', 'Greek'],
  ['en', 'English'],
  ['es-XX', 'Spanish'],
  ['et-EE', 'Estonian'],
  ['fa-IR', 'Persian'],
  ['fi-FI', 'Finnish'],
  ['fil-PH', 'Filipino'],
  ['fr-CA', 'French'],
  ['fr-FR', 'French'],
  ['gu-IN', 'Gujarati'],
  ['he-IL', 'Hebrew'],
  ['hi-IN', 'Hindi'],
  ['hr-HR', 'Croatian'],
  ['hu-HU', 'Hungarian'],
  ['id-ID', 'Indonesian'],
  ['is-IS', 'Icelandic'],
  ['it-IT', 'Italian'],
  ['ja-JP', 'Japanese'],
  ['kn-IN', 'Kannada'],
  ['ko-KR', 'Korean'],
  ['lt-LT', 'Lithuanian'],
  ['lv-LV', 'Latvian'],
  ['ml-IN', 'Malayalam'],
  ['mr-IN', 'Marathi'],
  ['nl-NL', 'Dutch'],
  ['no-NO', 'Norwegian'],
  ['pa-IN', 'Punjabi'],
  ['pl-PL', 'Polish'],
  ['pt-BR', 'Portuguese'],
  ['pt-PT', 'Portuguese'],
  ['ro-RO', 'Romanian'],
  ['ru-RU', 'Russian'],
  ['sk-SK', 'Slovak'],
  ['sl-SI', 'Slovenian'],
  ['sr-RS', 'Serbian'],
  ['sv-SE', 'Swedish'],
  ['sw-KE', 'Swahili'],
  ['sw-TZ', 'Swahili'],
  ['ta-IN', 'Tamil'],
  ['te-IN', 'Telugu'],
  ['th-TH', 'Thai'],
  ['tr-TR', 'Turkish'],
  ['uk-UA', 'Ukrainian'],
  ['ur-PK', 'Urdu'],
  ['vi-VN', 'Vietnamese'],
  ['zh-TW', 'Chinese'],
  ['zu-ZA', 'Zulu'],
]);

const TRANSLATEGEMMA_LANGUAGE_NAMES = new Map();
const CHAT_ROLES = Object.freeze(['system', 'user', 'assistant']);
const CHAT_ROLE_SET = new Set(CHAT_ROLES);

function assertSupportedChatRole(role, templateName, messageIndex) {
  if (!CHAT_ROLE_SET.has(role)) {
    const suffix = Number.isInteger(messageIndex) ? ` at message index ${messageIndex}` : '';
    throw new Error(
      `${templateName} formatter expects message role "${CHAT_ROLES.join('", "')}".${suffix}`
    );
  }
}

for (const [code, name] of TRANSLATEGEMMA_LANGUAGE_ENTRIES) {
  TRANSLATEGEMMA_LANGUAGE_NAMES.set(code, name);
  const rootCode = code.split('-')[0];
  if (!TRANSLATEGEMMA_LANGUAGE_NAMES.has(rootCode)) {
    TRANSLATEGEMMA_LANGUAGE_NAMES.set(rootCode, name);
  }
}

function normalizeChatRole(role) {
  if (typeof role !== 'string') return null;
  const normalized = role.trim().toLowerCase();
  if (normalized === 'system' || normalized === 'user' || normalized === 'assistant') {
    return normalized;
  }
  return null;
}

function normalizeTranslateLanguageCode(value) {
  if (typeof value !== 'string') return null;
  const normalized = value.trim().replace(/_/g, '-');
  return normalized || null;
}

function resolveTranslateLanguageName(value) {
  const normalized = normalizeTranslateLanguageCode(value);
  if (!normalized) return null;
  return TRANSLATEGEMMA_LANGUAGE_NAMES.get(normalized) ?? null;
}

function normalizeChatMessageContent(content) {
  if (typeof content === 'string') {
    return content;
  }
  if (Array.isArray(content) && content.length > 0) {
    const block = content[0];
    if (block && typeof block === 'object' && !Array.isArray(block)) {
      if (block.type === 'text') {
        return String(block.text ?? '');
      }
      if (block.type === 'image') {
        return '<start_of_image>';
      }
    }
    return content.map((entry) => String(entry ?? '')).join('\n');
  }
  if (content && typeof content === 'object' && !Array.isArray(content)) {
    if (content.type === 'text') return String(content.text ?? '');
    if (content.type === 'image') return '<start_of_image>';
  }
  return String(content ?? '');
}

function stripThinking(content) {
  if (typeof content !== 'string') {
    return String(content ?? '');
  }
  return content
    .replace(/<think>[\s\S]*?<\/think>/g, '')
    .replace(/<thinking>[\s\S]*?<\/thinking>/g, '')
    .replace(/<\|think\|>[\s\S]*?<\|\/think\|>/g, '')
    .replace(/<\|think\|>[\s\S]*$/g, '')
    .trim();
}

function renderGemma4ContentPart(part, role) {
  if (part == null) {
    return '';
  }
  if (typeof part === 'string') {
    return role === 'assistant' ? stripThinking(part) : part.trim();
  }
  if (typeof part !== 'object' || Array.isArray(part)) {
    return String(part);
  }

  if (part.type === 'text') {
    return role === 'assistant'
      ? stripThinking(String(part.text ?? ''))
      : String(part.text ?? '').trim();
  }
  if (part.type === 'image') {
    return '\n\n<|image|>\n\n';
  }
  if (part.type === 'audio') {
    return '<|audio|>';
  }
  if (part.type === 'video') {
    return '\n\n<|video|>\n\n';
  }
  return String(part.text ?? part.content ?? '');
}

function renderGemma4MessageContent(content, role) {
  if (Array.isArray(content)) {
    return content.map((part) => renderGemma4ContentPart(part, role)).join('');
  }
  return renderGemma4ContentPart(content, role);
}

function formatTurnBased(messages) {
  // Turn-based format: <start_of_turn>role\ncontent<end_of_turn>
  const parts = [];
  let systemContent = '';

  for (const [index, message] of messages.entries()) {
    const role = normalizeChatRole(message?.role);
    assertSupportedChatRole(role, 'Turn-based', index);
    if (role === 'system') {
      const content = normalizeChatMessageContent(message?.content);
      systemContent += (systemContent ? '\n\n' : '') + content;
    }
  }

  for (const [index, message] of messages.entries()) {
    const role = normalizeChatRole(message?.role);
    assertSupportedChatRole(role, 'Turn-based', index);
    if (role === 'system') continue;

    if (role === 'user') {
      const content = normalizeChatMessageContent(message?.content);
      const fullContent = systemContent ? `${systemContent}\n\n${content}` : content;
      systemContent = '';
      parts.push(`<start_of_turn>user\n${fullContent}<end_of_turn>\n`);
      continue;
    }
    if (role === 'assistant') {
      const content = normalizeChatMessageContent(message?.content);
      parts.push(`<start_of_turn>model\n${content}<end_of_turn>\n`);
    }
  }

  parts.push('<start_of_turn>model\n');

  return parts.join('');
}

function formatGemma4(messages, options) {
  const parts = ['<bos>'];
  for (const [index, message] of messages.entries()) {
    const role = normalizeChatRole(message?.role);
    assertSupportedChatRole(role, 'Gemma 4', index);
    const content = renderGemma4MessageContent(message?.content, role);
    if (role === 'system') {
      parts.push(`<|turn>system\n${content}<turn|>\n`);
      continue;
    }
    if (role === 'user') {
      parts.push(`<|turn>user\n${content}<turn|>\n`);
      continue;
    }
    if (role === 'assistant') {
      parts.push(`<|turn>model\n${content}<turn|>\n`);
    }
  }
  if (options?.thinking === true) {
    parts.push('<|turn>model\n<|think|>\n');
  } else {
    parts.push('<|turn>model\n');
  }
  return parts.join('');
}

function formatMessagesWithRoleWrap(messages, templateName, roleWrappers, options = {}) {
  const { prefix = '', suffix = '', beforeMessage = null } = options;
  const parts = prefix ? [prefix] : [];
  for (const [index, message] of messages.entries()) {
    const role = normalizeChatRole(message?.role);
    assertSupportedChatRole(role, templateName, index);
    if (beforeMessage) beforeMessage(role, index);
    const content = normalizeChatMessageContent(message?.content);
    const wrapper = roleWrappers[role];
    if (wrapper) parts.push(wrapper(content));
  }
  parts.push(suffix);
  return parts.join('');
}

function formatHeaderBased(messages) {
  return formatMessagesWithRoleWrap(messages, 'Header-based', {
    system: (c) => `<|start_header_id|>system<|end_header_id|>\n\n${c}<|eot_id|>`,
    user: (c) => `<|start_header_id|>user<|end_header_id|>\n\n${c}<|eot_id|>`,
    assistant: (c) => `<|start_header_id|>assistant<|end_header_id|>\n\n${c}<|eot_id|>`,
  }, {
    prefix: '<|begin_of_text|>',
    suffix: '<|start_header_id|>assistant<|end_header_id|>\n\n',
  });
}

function formatChannelBased(messages) {
  return formatMessagesWithRoleWrap(messages, 'Channel-based', {
    system: (c) => `<|start|>system<|message|>${c}<|end|>`,
    user: (c) => `<|start|>user<|message|>${c}<|end|>`,
    assistant: (c) => `<|start|>assistant<|channel|>final<|message|>${c}<|end|>`,
  }, {
    suffix: '<|start|>assistant<|channel|>final<|message|>',
  });
}

function formatChatML(messages) {
  return formatMessagesWithRoleWrap(messages, 'ChatML', {
    system: (c) => `<|im_start|>system\n${c}<|im_end|>\n`,
    user: (c) => `<|im_start|>user\n${c}<|im_end|>\n`,
    assistant: (c) => `<|im_start|>assistant\n${c}<|im_end|>\n`,
  }, {
    suffix: '<|im_start|>assistant\n',
  });
}

function formatQwen(messages) {
  return formatMessagesWithRoleWrap(messages, 'Qwen', {
    system: (c) => `<|im_start|>system\n${c}<|im_end|>\n`,
    user: (c) => `<|im_start|>user\n${c}<|im_end|>\n`,
    assistant: (c) => `<|im_start|>assistant\n${c}<|im_end|>\n`,
  }, {
    suffix: '<|im_start|>assistant\n<think>\n\n</think>\n\n',
    beforeMessage: (role, index) => {
      if (role === 'system' && index !== 0) {
        throw new Error('Qwen template requires any system message to appear first.');
      }
    },
  });
}

function formatTranslateGemmaUserPrompt(content) {
  if (!Array.isArray(content) || content.length !== 1) {
    throw new Error(
      'TranslateGemma template requires user content as an array with exactly one item.'
    );
  }
  const block = content[0];
  if (!block || typeof block !== 'object' || Array.isArray(block)) {
    throw new Error(
      'TranslateGemma template requires content[0] to be an object with type/source_lang_code/target_lang_code.'
    );
  }

  const sourceLangCode = normalizeTranslateLanguageCode(block.source_lang_code);
  const targetLangCode = normalizeTranslateLanguageCode(block.target_lang_code);
  if (!sourceLangCode) {
    throw new Error('TranslateGemma template requires source_lang_code.');
  }
  if (!targetLangCode) {
    throw new Error('TranslateGemma template requires target_lang_code.');
  }

  const sourceLang = resolveTranslateLanguageName(sourceLangCode);
  const targetLang = resolveTranslateLanguageName(targetLangCode);
  if (!sourceLang) {
    throw new Error(`TranslateGemma template: unsupported source_lang_code "${sourceLangCode}".`);
  }
  if (!targetLang) {
    throw new Error(`TranslateGemma template: unsupported target_lang_code "${targetLangCode}".`);
  }

  const instructionPrefix = (
    `You are a professional ${sourceLang} (${sourceLangCode}) to ${targetLang} (${targetLangCode}) translator. ` +
    `Your goal is to accurately convey the meaning and nuances of the original ${sourceLang} text while ` +
    `adhering to ${targetLang} grammar, vocabulary, and cultural sensitivities.\n`
  );

  if (block.type === 'text') {
    return (
      instructionPrefix +
      `Produce only the ${targetLang} translation, without any additional explanations or ` +
      `commentary. Please translate the following ${sourceLang} text into ${targetLang}:\n\n\n` +
      String(block.text ?? '').trim()
    );
  }

  if (block.type === 'image') {
    return (
      instructionPrefix +
      `Please translate the ${sourceLang} text in the provided image into ${targetLang}. ` +
      `Produce only the ${targetLang} translation, without any additional explanations, ` +
      `alternatives or commentary. Focus only on the text, do not output where the text is located, ` +
      `surrounding objects or any other explanation about the picture. Ignore symbols, pictogram, and ` +
      `arrows!\n\n\n<start_of_image>`
    );
  }

  throw new Error('TranslateGemma template only supports user content type "text" or "image".');
}

function formatTranslateGemma(messages) {
  if (!Array.isArray(messages) || messages.length === 0) {
    throw new Error('TranslateGemma template requires at least one message.');
  }
  const firstRole = normalizeChatRole(messages[0]?.role);
  assertSupportedChatRole(firstRole, 'TranslateGemma', 0);
  if (firstRole !== 'user') {
    throw new Error('TranslateGemma template requires the conversation to start with a user message.');
  }

  const parts = [];
  for (let index = 0; index < messages.length; index++) {
    const message = messages[index];
    const role = normalizeChatRole(message?.role);
    assertSupportedChatRole(role, 'TranslateGemma', index);
    if ((role === 'user') !== (index % 2 === 0)) {
      throw new Error('TranslateGemma template requires alternating user/assistant roles.');
    }

    if (role === 'assistant') {
      if (message?.content == null || typeof message.content !== 'string') {
        throw new Error('TranslateGemma template requires assistant messages to use string content.');
      }
      parts.push(`<start_of_turn>model\n${message.content.trim()}<end_of_turn>\n`);
      continue;
    }

    if (role === 'user') {
      const userPrompt = formatTranslateGemmaUserPrompt(message?.content);
      parts.push(`<start_of_turn>user\n${userPrompt}<end_of_turn>\n`);
      continue;
    }

    throw new Error('TranslateGemma template only supports user and assistant roles.');
  }

  parts.push('<start_of_turn>model\n');
  return parts.join('');
}

function formatPlaintext(messages) {
  // Simple plaintext format for unknown templates
  return messages
    .map((message) => {
      const role = normalizeChatRole(message?.role);
      const content = normalizeChatMessageContent(message?.content);
      if (role === 'system') return `System: ${content}`;
      if (role === 'user') return `User: ${content}`;
      if (role === 'assistant') return `Assistant: ${content}`;
      return content;
    })
    .join('\n') + '\nAssistant:';
}

// Template type to formatter mapping.
// Add new template types here rather than adding switch cases.
const CHAT_FORMATTERS = {
  'gemma': formatTurnBased,
  'gemma4': formatGemma4,
  'llama3': formatHeaderBased,
  'gpt-oss': formatChannelBased,
  'chatml': formatChatML,
  'qwen': formatQwen,
  'translategemma': formatTranslateGemma,
};

export function formatChatMessages(messages, templateType, options) {
  if (!Array.isArray(messages)) {
    throw new Error('formatChatMessages expects an array of messages.');
  }
  if (templateType == null) {
    return formatPlaintext(messages);
  }
  const formatter = CHAT_FORMATTERS[templateType];
  if (formatter) {
    return formatter(messages, options);
  }
  throw new Error(`Unrecognized chat template type: ${templateType}`);
}

export const formatGemmaChat = formatTurnBased;
export const formatGemma4Chat = formatGemma4;
export const formatLlama3Chat = formatHeaderBased;
export const formatGptOssChat = formatChannelBased;
export const formatQwenChat = formatQwen;
export const formatTranslateGemmaChat = formatTranslateGemma;
