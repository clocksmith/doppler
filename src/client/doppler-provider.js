// Types and interfaces
export {
  DOPPLER_PROVIDER_VERSION,
  DopplerCapabilities,
} from './doppler-provider/types.js';

// Model management
export {
  initDoppler,
  loadModel,
  unloadModel,
  loadLoRAAdapter,
  activateLoRAFromTrainingOutput,
  unloadLoRAAdapter,
  getActiveLoRA,
  getAvailableModels,
  destroyDoppler,
  getPipeline,
  getCurrentModelId,
} from './doppler-provider/model-manager.js';

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
} from './doppler-provider/generation.js';

// Main provider
export { DopplerProvider, default } from './doppler-provider/provider.js';
