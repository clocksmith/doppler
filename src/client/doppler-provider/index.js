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

// Runtime config
export {
  getRuntimeConfig,
  setRuntimeConfig,
  resetRuntimeConfig,
} from '../../config/runtime.js';

// Conversion
export {
  ConvertStage,
  isConversionSupported,
  createRemoteModelSources,
  convertModel,
  pickModelFiles,
} from '../../browser/browser-converter.js';

// Harness
export {
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

// Main provider
export { DopplerProvider, default } from './provider.js';
