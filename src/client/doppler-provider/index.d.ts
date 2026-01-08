// Types and interfaces
export {
  DOPPLER_PROVIDER_VERSION,
  DopplerCapabilities,
  type TextModelConfig,
  type InferredAttentionParams,
  type ModelEstimate,
  type LoadProgressEvent,
  type GenerateOptions,
  type ChatMessage,
  type ChatResponse,
  type DopplerCapabilitiesType,
  type DopplerProviderInterface,
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
  formatChatMessages,
  dopplerChat,
} from './generation.js';

// Main provider
export { DopplerProvider, default } from './provider.js';
