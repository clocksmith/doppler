// Re-export everything from the modular implementation
export {
  // Version
  DOPPLER_PROVIDER_VERSION,

  // Types
  type TextModelConfig,
  type InferredAttentionParams,
  type ModelEstimate,
  type LoadProgressEvent,
  type GenerateOptions,
  type ChatMessage,
  type ChatResponse,
  type DopplerCapabilitiesType,
  type DopplerProviderInterface,

  // Capability flags
  DopplerCapabilities,

  // Model management
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

  // Generation
  generate,
  prefillKV,
  generateWithPrefixKV,
  formatGemmaChat,
  formatLlama3Chat,
  formatGptOssChat,
  formatChatMessages,
  buildChatPrompt,
  dopplerChat,

  // Main provider
  DopplerProvider,
  default,
} from './doppler-provider/index.js';
