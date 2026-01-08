import { log } from '../../debug/index.js';
import { getPipeline } from './model-manager.js';

export async function* generate(prompt, options = {}) {
  const pipeline = getPipeline();
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }

  const {
    maxTokens = 256,
    temperature = 0.7,
    topP = 0.9,
    topK = 40,
    stopSequences = [],
    onToken = null,
  } = options;

  for await (const token of pipeline.generate(prompt, {
    maxTokens,
    temperature,
    topP,
    topK,
    stopSequences,
  })) {
    if (onToken) onToken(token);
    yield token;
  }
}

export async function prefillKV(prompt, options = {}) {
  const pipeline = getPipeline();
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }
  const { onToken: _unused, ...pipelineOptions } = options;
  return pipeline.prefillKVOnly(prompt, pipelineOptions);
}

export async function* generateWithPrefixKV(prefix, prompt, options = {}) {
  const pipeline = getPipeline();
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }
  const { onToken, ...pipelineOptions } = options;
  for await (const token of pipeline.generateWithPrefixKV(prefix, prompt, pipelineOptions)) {
    if (onToken) onToken(token);
    yield token;
  }
}

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

export function formatChatMessages(messages) {
  const pipeline = getPipeline();
  const isGemma = pipeline?.modelConfig?.isGemma3;
  const isLlama3 = pipeline?.modelConfig?.isLlama3Instruct;

  if (isGemma) {
    return formatGemmaChat(messages);
  } else if (isLlama3) {
    return formatLlama3Chat(messages);
  }

  return messages
    .map((m) => {
      if (m.role === 'system') return `System: ${m.content}`;
      if (m.role === 'user') return `User: ${m.content}`;
      if (m.role === 'assistant') return `Assistant: ${m.content}`;
      return m.content;
    })
    .join('\n') + '\nAssistant:';
}

export async function dopplerChat(messages, options = {}) {
  const pipeline = getPipeline();
  const prompt = formatChatMessages(messages);

  let promptTokens = 0;
  if (pipeline && pipeline.tokenizer) {
    try {
      const encoded = pipeline.tokenizer.encode(prompt);
      promptTokens = encoded.length;
    } catch (e) {
      log.warn('DopplerProvider', 'Failed to count prompt tokens', e);
    }
  }

  const tokens = [];
  for await (const token of generate(prompt, options)) {
    tokens.push(token);
  }

  return {
    content: tokens.join(''),
    usage: {
      promptTokens,
      completionTokens: tokens.length,
      totalTokens: promptTokens + tokens.length,
    },
  };
}
