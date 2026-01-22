// Re-export everything from the modular implementation
export {
  // Version
  DOPPLER_PROVIDER_VERSION,

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
