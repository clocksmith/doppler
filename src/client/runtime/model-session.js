import { getKernelCapabilities } from '../../gpu/device.js';
import { formatChatMessages } from '../../inference/pipelines/text/chat-format.js';
import {
  activateLoRAFromTrainingOutputForPipeline,
  getActiveLoRAForPipeline,
  loadLoRAAdapterForPipeline,
  unloadLoRAAdapterForPipeline,
} from './lora.js';

export function assertSupportedGenerationOptions(options = {}) {
  if (Array.isArray(options?.stopTokens) && options.stopTokens.length > 0) {
    throw new Error(
      'Doppler generate options do not support stopTokens on this surface. ' +
      'Use stopSequences instead.'
    );
  }
}

function countTokens(pipeline, text) {
  if (!text || typeof text !== 'string') return 0;
  try {
    return pipeline?.tokenizer?.encode(text)?.length ?? 0;
  } catch {
    return 0;
  }
}

function resolveChatPromptForUsage(pipeline, messages) {
  const templateType = pipeline?.manifest?.inference?.chatTemplate?.enabled === false
    ? null
    : (pipeline?.manifest?.inference?.chatTemplate?.type ?? null);
  try {
    return formatChatMessages(messages, templateType);
  } catch {
    return messages.map((message) => String(message?.content ?? '')).join('\n');
  }
}

async function collectText(iterable) {
  let output = '';
  for await (const token of iterable) {
    output += token;
  }
  return output;
}

export function createModelHandle(pipeline, resolved) {
  return {
    generate(prompt, options = {}) {
      assertSupportedGenerationOptions(options);
      return pipeline.generate(prompt, options);
    },
    async generateText(prompt, options = {}) {
      assertSupportedGenerationOptions(options);
      return collectText(pipeline.generate(prompt, options));
    },
    chat(messages, options = {}) {
      assertSupportedGenerationOptions(options);
      return pipeline.generate(messages, options);
    },
    async chatText(messages, options = {}) {
      assertSupportedGenerationOptions(options);
      const content = await collectText(pipeline.generate(messages, options));
      const promptText = resolveChatPromptForUsage(pipeline, messages);
      const promptTokens = countTokens(pipeline, promptText);
      const completionTokens = countTokens(pipeline, content);
      return {
        content,
        usage: {
          promptTokens,
          completionTokens,
          totalTokens: promptTokens + completionTokens,
        },
      };
    },
    async embed(prompt, options = {}) {
      return pipeline.embed(prompt, options);
    },
    async embedBatch(prompts, options = {}) {
      return pipeline.embedBatch(prompts, options);
    },
    async embedImage(args = {}) {
      return pipeline.embedImage(args);
    },
    async embedAudio(args = {}) {
      return pipeline.embedAudio(args);
    },
    async transcribeImage(args = {}) {
      return pipeline.transcribeImage(args);
    },
    async transcribeAudio(args = {}) {
      return pipeline.transcribeAudio(args);
    },
    async transcribeVideo(args = {}) {
      return pipeline.transcribeVideo(args);
    },
    get supportsEmbedding() {
      return pipeline.manifest?.modelType === 'embedding'
        || pipeline.manifest?.inference?.supportsEmbedding === true;
    },
    get supportsTranscription() {
      return pipeline.manifest?.inference?.supportsTranscription === true
        && pipeline.audioCapable === true;
    },
    get supportsVision() {
      return pipeline.manifest?.inference?.supportsVision === true
        && pipeline.visionCapable === true;
    },
    async loadLoRA(adapter) {
      return loadLoRAAdapterForPipeline(pipeline, adapter);
    },
    async activateLoRAFromTrainingOutput(trainingOutput) {
      return activateLoRAFromTrainingOutputForPipeline(pipeline, trainingOutput);
    },
    async unloadLoRA() {
      return unloadLoRAAdapterForPipeline(pipeline);
    },
    async unload() {
      await pipeline.unload();
    },
    get activeLoRA() {
      return getActiveLoRAForPipeline(pipeline);
    },
    get loaded() {
      return pipeline.isLoaded === true;
    },
    get modelId() {
      return resolved.modelId;
    },
    get manifest() {
      return pipeline.manifest;
    },
    get deviceInfo() {
      return getKernelCapabilities()?.adapterInfo ?? null;
    },
    advanced: {
      prefillKV(prompt, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.prefillKVOnly(prompt, options);
      },
      prefillWithLogits(prompt, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.prefillWithLogits(prompt, options);
      },
      decodeStepLogits(currentIds, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.decodeStepLogits(currentIds, options);
      },
      generateWithPrefixKV(prefix, prompt, options = {}) {
        assertSupportedGenerationOptions(options);
        return pipeline.generateWithPrefixKV(prefix, prompt, options);
      },
    },
  };
}
