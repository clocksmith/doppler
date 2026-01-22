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
  destroyDoppler,
  getPipeline,
  getCurrentModelId,
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
