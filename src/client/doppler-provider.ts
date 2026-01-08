/**
 * DOPPLER Provider - LLM Client Integration
 * Registers DOPPLER as a local WebGPU option in llm-client.js
 *
 * This is a facade module that re-exports from the modular implementation.
 * For new code, prefer importing from './doppler-provider/index.js' directly.
 */

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
