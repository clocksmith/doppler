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
  type DopplerProviderRuntime,
  type DopplerProviderConversion,
  type DopplerProviderBench,
  type RuntimeConfigSchema,
  type ConvertStageType,
  type ConvertProgress,
  type ConvertOptions,
  type ShardInfo,
  type TensorLocation,
  type RDRRManifest,
  type TensorSource,
  type HttpTensorSourceOptions,
  type BrowserHarnessOptions,
  type BrowserSuiteOptions,
  type BrowserManifest,
  type BrowserHarnessResult,
  type BrowserSuiteResult,
  type BrowserManifestResult,
  type BrowserSuite,
  type SuiteSummary,
  type SuiteTestResult,
  type RuntimeConfigLoadOptions,

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
  formatGptOssChat,
  formatChatMessages,
  buildChatPrompt,
  dopplerChat,

  // Runtime config
  getRuntimeConfig,
  setRuntimeConfig,
  resetRuntimeConfig,

  // Conversion
  ConvertStage,
  isConversionSupported,
  createRemoteModelSources,
  convertModel,
  pickModelFiles,

  // Harness
  loadRuntimeConfigFromUrl,
  applyRuntimeConfigFromUrl,
  loadRuntimePreset,
  applyRuntimePreset,
  initializeBrowserHarness,
  saveBrowserReport,
  runBrowserHarness,
  runBrowserSuite,
  runBrowserManifest,

  // Main provider
  DopplerProvider,
  default,
} from './doppler-provider/index.js';
