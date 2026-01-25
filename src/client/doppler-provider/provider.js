import { DopplerCapabilities } from './types.js';
import {
  initDoppler,
  loadModel,
  unloadModel,
  loadLoRAAdapter,
  unloadLoRAAdapter,
  getActiveLoRA,
  getAvailableModels,
  getDopplerStorageInfo,
  getPipeline,
  getCurrentModelId,
  extractTextModelConfig,
  readOPFSFile,
  writeOPFSFile,
  fetchArrayBuffer,
  destroyDoppler,
} from './model-manager.js';
import { getRuntimeConfig, setRuntimeConfig, resetRuntimeConfig } from '../../config/runtime.js';
import {
  ConvertStage,
  isConversionSupported,
  createRemoteModelSources,
  convertModel,
  pickModelFiles,
} from '../../browser/browser-converter.js';
import {
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimePreset,
  applyRuntimePreset,
  initializeBrowserHarness,
  saveBrowserReport,
  runBrowserHarness,
  runBrowserSuite,
  runBrowserManifest,
} from '../../inference/browser-harness.js';
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

  async getDopplerStorageInfo() {
    return getDopplerStorageInfo();
  },

  getPipeline() {
    return getPipeline();
  },

  getCurrentModelId() {
    return getCurrentModelId();
  },

  extractTextModelConfig(manifest) {
    return extractTextModelConfig(manifest);
  },

  async readOPFSFile(path) {
    return readOPFSFile(path);
  },

  async writeOPFSFile(path, data) {
    return writeOPFSFile(path, data);
  },

  async fetchArrayBuffer(url) {
    return fetchArrayBuffer(url);
  },

  runtime: {
    getRuntimeConfig,
    setRuntimeConfig,
    resetRuntimeConfig,
  },

  conversion: {
    ConvertStage,
    isConversionSupported,
    createRemoteModelSources,
    convertModel,
    pickModelFiles,
  },

  bench: {
    loadRuntimeConfigFromUrl,
    applyRuntimeConfigFromUrl,
    loadRuntimePreset,
    applyRuntimePreset,
    initializeBrowserHarness,
    saveBrowserReport,
    runBrowserHarness,
    runBrowserSuite,
    runBrowserManifest,
  },

  async destroy() {
    return destroyDoppler();
  },
};

export default DopplerProvider;
