/**
 * DOPPLER Generation Module
 * Text generation, streaming, and chat formatting logic.
 */

import type { KVCacheSnapshot } from '../../inference/pipeline.js';
import { log } from '../../debug/index.js';
import { getPipeline } from './model-manager.js';
import type { GenerateOptions, ChatMessage, ChatResponse } from './types.js';

/**
 * Generate text completion
 * @param prompt - Input prompt
 * @param options - Generation options
 * @returns Token stream
 */
export async function* generate(prompt: string, options: GenerateOptions = {}): AsyncGenerator<string> {
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

/**
 * Prefill KV cache with a prompt (for prefix caching)
 */
export async function prefillKV(prompt: string, options: GenerateOptions = {}): Promise<KVCacheSnapshot> {
  const pipeline = getPipeline();
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }
  // Extract pipeline-compatible options (exclude provider-specific onToken)
  const { onToken: _unused, ...pipelineOptions } = options;
  return pipeline.prefillKVOnly(prompt, pipelineOptions);
}

/**
 * Generate with a pre-filled KV cache prefix
 */
export async function* generateWithPrefixKV(
  prefix: KVCacheSnapshot,
  prompt: string,
  options: GenerateOptions = {}
): AsyncGenerator<string> {
  const pipeline = getPipeline();
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }
  // Extract pipeline-compatible options (exclude provider-specific onToken)
  const { onToken, ...pipelineOptions } = options;
  for await (const token of pipeline.generateWithPrefixKV(prefix, prompt, pipelineOptions)) {
    if (onToken) onToken(token);
    yield token;
  }
}

/**
 * Format chat messages for Gemma models.
 *
 * Gemma uses: <start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n{content}<end_of_turn>\n
 * Gemma does NOT support system role - system instructions are merged into first user message.
 *
 * @param messages - Chat messages
 * @returns Formatted prompt string
 */
export function formatGemmaChat(messages: ChatMessage[]): string {
  const parts: string[] = [];
  let systemContent = '';

  // Extract system content (will be prepended to first user message)
  for (const m of messages) {
    if (m.role === 'system') {
      systemContent += (systemContent ? '\n\n' : '') + m.content;
    }
  }

  // Format user/assistant turns
  for (const m of messages) {
    if (m.role === 'system') continue; // Already extracted

    if (m.role === 'user') {
      // Prepend system content to first user message
      const content = systemContent
        ? `${systemContent}\n\n${m.content}`
        : m.content;
      systemContent = ''; // Only prepend once
      parts.push(`<start_of_turn>user\n${content}<end_of_turn>\n`);
    } else if (m.role === 'assistant') {
      parts.push(`<start_of_turn>model\n${m.content}<end_of_turn>\n`);
    }
  }

  // Add model turn prefix for generation
  parts.push('<start_of_turn>model\n');

  return parts.join('');
}

/**
 * Format chat messages for Llama 3 instruct models.
 *
 * @param messages - Chat messages
 * @returns Formatted prompt string
 */
export function formatLlama3Chat(messages: ChatMessage[]): string {
  const parts: string[] = ['<|begin_of_text|>'];

  for (const m of messages) {
    if (m.role === 'system') {
      parts.push(`<|start_header_id|>system<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    } else if (m.role === 'user') {
      parts.push(`<|start_header_id|>user<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    } else if (m.role === 'assistant') {
      parts.push(`<|start_header_id|>assistant<|end_header_id|>\n\n${m.content}<|eot_id|>`);
    }
  }

  // Add assistant turn prefix for generation
  parts.push('<|start_header_id|>assistant<|end_header_id|>\n\n');

  return parts.join('');
}

/**
 * Format chat messages based on model type.
 *
 * @param messages - Chat messages
 * @returns Formatted prompt string
 */
export function formatChatMessages(messages: ChatMessage[]): string {
  const pipeline = getPipeline();
  // Check model type from pipeline config
  const isGemma = pipeline?.modelConfig?.isGemma3;
  const isLlama3 = pipeline?.modelConfig?.isLlama3Instruct;

  if (isGemma) {
    return formatGemmaChat(messages);
  } else if (isLlama3) {
    return formatLlama3Chat(messages);
  }

  // Generic fallback format
  return messages
    .map((m) => {
      if (m.role === 'system') return `System: ${m.content}`;
      if (m.role === 'user') return `User: ${m.content}`;
      if (m.role === 'assistant') return `Assistant: ${m.content}`;
      return m.content;
    })
    .join('\n') + '\nAssistant:';
}

/**
 * Chat completion (matches LLM client interface)
 * @param messages - Chat messages
 * @param options - Generation options
 */
export async function dopplerChat(messages: ChatMessage[], options: GenerateOptions = {}): Promise<ChatResponse> {
  const pipeline = getPipeline();
  // Format messages using model-aware template
  const prompt = formatChatMessages(messages);

  // Count prompt tokens using pipeline's tokenizer
  let promptTokens = 0;
  if (pipeline && pipeline.tokenizer) {
    try {
      const encoded = pipeline.tokenizer.encode(prompt);
      promptTokens = encoded.length;
    } catch (e) {
      log.warn('DopplerProvider', 'Failed to count prompt tokens', e);
    }
  }

  const tokens: string[] = [];
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
