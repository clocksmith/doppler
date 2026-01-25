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
  type DopplerProviderRuntime,
  type DopplerProviderConversion,
  type DopplerProviderBench,
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
export type { RuntimeConfigSchema } from '../../config/schema/index.js';

// Conversion
export {
  ConvertStage,
  isConversionSupported,
  createRemoteModelSources,
  convertModel,
  pickModelFiles,
} from '../../browser/browser-converter.js';
export type {
  ConvertStageType,
  ConvertProgress,
  ConvertOptions,
  ShardInfo,
  TensorLocation,
  RDRRManifest,
} from '../../browser/browser-converter.js';
export type { TensorSource } from '../../browser/tensor-source-file.js';
export type { HttpTensorSourceOptions } from '../../browser/tensor-source-http.js';

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
export type {
  BrowserHarnessOptions,
  BrowserSuiteOptions,
  BrowserManifest,
  BrowserHarnessResult,
  BrowserSuiteResult,
  BrowserManifestResult,
  BrowserSuite,
  SuiteSummary,
  SuiteTestResult,
  RuntimeConfigLoadOptions,
} from '../../inference/browser-harness.js';

// Main provider
export { DopplerProvider, default } from './provider.js';
