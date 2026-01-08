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
  getDopplerStorageInfo,
  destroyDoppler,
  getPipeline,
  getCurrentModelId,
  extractTextModelConfig,
  readOPFSFile,
  writeOPFSFile,
  fetchArrayBuffer,

  // Generation
  generate,
  prefillKV,
  generateWithPrefixKV,
  formatGemmaChat,
  formatLlama3Chat,
  formatChatMessages,
  dopplerChat,

  // Main provider
  DopplerProvider,
  default,
} from './doppler-provider/index.js';
