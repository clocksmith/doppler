import { DopplerCapabilities } from './types.js';
import {
  initDoppler,
  loadModel,
  unloadModel,
  loadLoRAAdapter,
  unloadLoRAAdapter,
  getActiveLoRA,
  getAvailableModels,
  getPipeline,
  getCurrentModelId,
  destroyDoppler,
} from './model-manager.js';
import {
  generate,
  prefillKV,
  generateWithPrefixKV,
  buildChatPrompt,
  dopplerChat,
} from './generation.js';

export const DopplerProvider = {
  name: 'doppler',
  displayName: 'DOPPLER',
  isLocal: true,

  async init() {
    return initDoppler();
  },

  async loadModel(modelId, modelUrl, onProgress, localPath) {
    return loadModel(modelId, modelUrl ?? null, onProgress ?? null, localPath ?? null);
  },

  async unloadModel() {
    return unloadModel();
  },

  async chat(messages, options) {
    return dopplerChat(messages, options);
  },

  async *stream(messages, options) {
    const prompt = buildChatPrompt(messages, options);
    for await (const token of generate(prompt, { ...options, useChatTemplate: false })) {
      yield token;
    }
  },

  async prefillKV(prompt, options) {
    return prefillKV(prompt, options);
  },

  async *generateWithPrefixKV(prefix, prompt, options) {
    for await (const token of generateWithPrefixKV(prefix, prompt, options)) {
      yield token;
    }
  },

  async loadLoRAAdapter(adapter) {
    return loadLoRAAdapter(adapter);
  },

  async unloadLoRAAdapter() {
    return unloadLoRAAdapter();
  },

  getActiveLoRA() {
    return getActiveLoRA();
  },

  getCapabilities() {
    return DopplerCapabilities;
  },

  async getModels() {
    return getAvailableModels();
  },

  async getAvailableModels() {
    return getAvailableModels();
  },

  getPipeline() {
    return getPipeline();
  },

  getCurrentModelId() {
    return getCurrentModelId();
  },

  async destroy() {
    return destroyDoppler();
  },
};

export default DopplerProvider;
