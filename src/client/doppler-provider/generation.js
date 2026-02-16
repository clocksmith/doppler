import { log } from '../../debug/index.js';
import {
  formatChatMessages as formatChatMessagesForTemplate,
  formatGemmaChat,
  formatLlama3Chat,
  formatGptOssChat,
} from '../../inference/pipelines/text/chat-format.js';
import { getRuntimeConfig } from '../../config/runtime.js';
import { DEFAULT_CHAT_TEMPLATE_CONFIG } from '../../config/schema/index.js';
import { getPipeline } from './model-manager.js';

export { formatGemmaChat, formatLlama3Chat, formatGptOssChat };

function resolveChatTemplate(pipeline, options) {
  const override = options?.useChatTemplate;
  const runtimeEnabled = pipeline?.runtimeConfig?.inference?.chatTemplate?.enabled;
  const modelEnabled = pipeline?.modelConfig?.chatTemplateEnabled;
  const enabled = override ?? runtimeEnabled ?? modelEnabled ?? DEFAULT_CHAT_TEMPLATE_CONFIG.enabled;
  const type = pipeline?.modelConfig?.chatTemplateType ?? null;
  return { enabled, type };
}

export async function* generate(prompt, options = {}) {
  const pipeline = getPipeline();
  if (!pipeline) {
    throw new Error('No model loaded. Call loadModel() first.');
  }

  const runtimeConfig = pipeline.runtimeConfig ?? getRuntimeConfig();
  const samplingDefaults = runtimeConfig.inference.sampling;
  const maxTokensDefault = runtimeConfig.inference.batching.maxTokens;

  const maxTokens = options.maxTokens ?? maxTokensDefault;
  const temperature = options.temperature ?? samplingDefaults.temperature;
  const topP = options.topP ?? samplingDefaults.topP;
  const topK = options.topK ?? samplingDefaults.topK;
  const stopSequences = options.stopSequences ?? [];
  const useChatTemplate = options.useChatTemplate;
  const onToken = options.onToken ?? null;

  for await (const token of pipeline.generate(prompt, {
    maxTokens,
    temperature,
    topP,
    topK,
    stopSequences,
    useChatTemplate,
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

export function formatChatMessages(messages, templateType) {
  const pipeline = getPipeline();
  const resolvedType = templateType ?? pipeline?.modelConfig?.chatTemplateType ?? null;
  return formatChatMessagesForTemplate(messages, resolvedType);
}

export function buildChatPrompt(messages, options = {}) {
  const pipeline = getPipeline();
  const { enabled, type } = resolveChatTemplate(pipeline, options);
  return formatChatMessagesForTemplate(messages, enabled ? type : null);
}

export async function dopplerChat(messages, options = {}) {
  const pipeline = getPipeline();
  const prompt = buildChatPrompt(messages, options);

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
  for await (const token of generate(prompt, { ...options, useChatTemplate: false })) {
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
