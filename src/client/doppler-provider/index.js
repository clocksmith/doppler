// Types and interfaces
export {
  DOPPLER_PROVIDER_VERSION,
  DopplerCapabilities,
} from './types.js';

// Model management
export {
  initDoppler,
  loadModel,
  unloadModel,
  loadLoRAAdapter,
  unloadLoRAAdapter,
  getActiveLoRA,
  getAvailableModels,
  getDopplerStorageInfo,
  destroyDoppler,
  getPipeline,
  getCurrentModelId,
  extractTextModelConfig,
  readOPFSFile,
  writeOPFSFile,
  fetchArrayBuffer,
} from './model-manager.js';

// Generation
export {
  generate,
  prefillKV,
  generateWithPrefixKV,
  formatGemmaChat,
  formatLlama3Chat,
  formatGptOssChat,
  formatChatMessages,
  buildChatPrompt,
  dopplerChat,
} from './generation.js';

// Main provider
export { DopplerProvider, default } from './provider.js';
