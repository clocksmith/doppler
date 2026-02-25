function normalizeRequiredString(value, fieldName) {
  const normalized = String(value ?? '').trim();
  if (!normalized) {
    throw new Error(`Translate request is missing required field "${fieldName}".`);
  }
  return normalized;
}

function readTranslateContentBlock(request) {
  const messages = Array.isArray(request?.messages) ? request.messages : null;
  if (!messages || messages.length !== 1) {
    throw new Error('Translate request must provide exactly one user message.');
  }
  const message = messages[0];
  if (message?.role !== 'user') {
    throw new Error('Translate request message role must be "user".');
  }
  const content = Array.isArray(message.content) ? message.content : null;
  if (!content || content.length !== 1) {
    throw new Error('Translate request user message must contain exactly one content item.');
  }
  const block = content[0];
  if (!block || typeof block !== 'object' || Array.isArray(block)) {
    throw new Error('Translate request content item must be an object.');
  }
  return block;
}

export function createTranslateTextRequest(text, sourceLangCode, targetLangCode) {
  return {
    messages: [
      {
        role: 'user',
        content: [
          {
            type: 'text',
            source_lang_code: normalizeRequiredString(sourceLangCode, 'source_lang_code'),
            target_lang_code: normalizeRequiredString(targetLangCode, 'target_lang_code'),
            text: String(text ?? ''),
          },
        ],
      },
    ],
  };
}

export function toTranslateTemplateLangCode(code) {
  return normalizeRequiredString(code, 'language_code').replace(/_/g, '-');
}

export function extractTranslateTextFields(request) {
  const content = readTranslateContentBlock(request);
  if (content.type !== 'text') {
    throw new Error(`Unsupported translate content type "${String(content.type)}". Expected "text".`);
  }
  return {
    sourceLangCode: normalizeRequiredString(content.source_lang_code, 'source_lang_code'),
    targetLangCode: normalizeRequiredString(content.target_lang_code, 'target_lang_code'),
    text: String(content.text ?? ''),
  };
}

export function buildTranslatePromptFromRequest(request, resolveLanguageName = null) {
  const {
    sourceLangCode,
    targetLangCode,
    text,
  } = extractTranslateTextFields(request);

  const languageNameResolver = typeof resolveLanguageName === 'function'
    ? resolveLanguageName
    : (code) => code;

  const sourceName = String(languageNameResolver(sourceLangCode) || sourceLangCode);
  const targetName = String(languageNameResolver(targetLangCode) || targetLangCode);
  const sourceCodeLabel = toTranslateTemplateLangCode(sourceLangCode);
  const targetCodeLabel = toTranslateTemplateLangCode(targetLangCode);

  return [
    `You are a professional ${sourceName} (${sourceCodeLabel}) to ${targetName} (${targetCodeLabel}) translator. ` +
      `Your goal is to accurately convey the meaning and nuances of the original ${sourceName} text while ` +
      `adhering to ${targetName} grammar, vocabulary, and cultural sensitivities.`,
    '',
    `Produce only the ${targetName} translation, without any additional explanations or commentary. ` +
      `Please translate the following ${sourceName} text into ${targetName}:`,
    '',
    '',
    text,
  ].join('\n');
}
