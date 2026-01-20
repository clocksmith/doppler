
// Chat template formatters keyed by template type.
// Template types are stored in manifest.inference.chatTemplate.type.

function formatTurnBased(messages) {
  // Turn-based format: <start_of_turn>role\ncontent<end_of_turn>
  const parts = [];
  let systemContent = '';

  for (const m of messages) {
    if (m.role === 'system') {
      systemContent += (systemContent ? '\n\n' : '') + m.content;
    }
  }

  for (const m of messages) {
    if (m.role === 'system') continue;

    if (m.role === 'user') {
      const content = systemContent
        ? `${systemContent}\n\n${m.content}`
        : m.content;
      systemContent = '';
      parts.push(`<start_of_turn>user\n${content}<end_of_turn>\n`);
    } else if (m.role === 'assistant') {
      parts.push(`<start_of_turn>model\n${m.content}<end_of_turn>\n`);
    }
  }

  parts.push('<start_of_turn>model\n');

  return parts.join('');
}


function formatHeaderBased(messages) {
  // Header-based format: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
  const parts = ['<|begin_of_text|>'];

  for (const m of messages) {
    if (m.role === 'system') {
      parts.push(`<|start_header_id|>system<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    } else if (m.role === 'user') {
      parts.push(`<|start_header_id|>user<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    } else if (m.role === 'assistant') {
      parts.push(`<|start_header_id|>assistant<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    }
  }

  parts.push('<|start_header_id|>assistant<|end_header_id|>\n\n');

  return parts.join('');
}


function formatChannelBased(messages) {
  // Channel-based format: <|start|>role<|channel|>channel<|message|>content<|end|>
  const parts = [];

  for (const m of messages) {
    if (m.role === 'system') {
      parts.push(`<|start|>system<|message|>${m.content}<|end|>`);
    } else if (m.role === 'user') {
      parts.push(`<|start|>user<|message|>${m.content}<|end|>`);
    } else if (m.role === 'assistant') {
      parts.push(`<|start|>assistant<|channel|>final<|message|>${m.content}<|end|>`);
    }
  }

  parts.push('<|start|>assistant<|channel|>final<|message|>');

  return parts.join('');
}


function formatPlaintext(messages) {
  // Simple plaintext format for unknown templates
  return messages
    .map((m) => {
      if (m.role === 'system') return `System: ${m.content}`;
      if (m.role === 'user') return `User: ${m.content}`;
      if (m.role === 'assistant') return `Assistant: ${m.content}`;
      return m.content;
    })
    .join('\n') + '\nAssistant:';
}

// Template type to formatter mapping.
// Add new template types here rather than adding switch cases.
const CHAT_FORMATTERS = {
  'gemma': formatTurnBased,
  'llama3': formatHeaderBased,
  'gpt-oss': formatChannelBased,
};


export function formatChatMessages(messages, templateType) {
  const formatter = CHAT_FORMATTERS[templateType];
  if (formatter) {
    return formatter(messages);
  }
  return formatPlaintext(messages);
}

// Legacy exports for backwards compatibility
export const formatGemmaChat = formatTurnBased;
export const formatLlama3Chat = formatHeaderBased;
export const formatGptOssChat = formatChannelBased;
