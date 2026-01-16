


export function formatGemmaChat(messages) {
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


export function formatLlama3Chat(messages) {
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


export function formatGptOssChat(messages) {
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


export function formatChatMessages(messages, templateType) {
  switch (templateType) {
    case 'gemma':
      return formatGemmaChat(messages);
    case 'llama3':
      return formatLlama3Chat(messages);
    case 'gpt-oss':
      return formatGptOssChat(messages);
    default:
      return messages
        .map((m) => {
          if (m.role === 'system') return `System: ${m.content}`;
          if (m.role === 'user') return `User: ${m.content}`;
          if (m.role === 'assistant') return `Assistant: ${m.content}`;
          return m.content;
        })
        .join('\n') + '\nAssistant:';
  }
}
