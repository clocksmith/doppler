/**
 * DOPPLER Provider Types
 * Type definitions and interfaces for the DOPPLER LLM provider.
 */

import type { ExtensionBridgeClient } from '../../bridge/index.js';
import type { KVCacheSnapshot } from '../../inference/pipeline.js';
import type { LoRAManifest } from '../../adapters/lora-loader.js';
import type { RDRRManifest } from '../../storage/rdrr-format.js';

export const DOPPLER_PROVIDER_VERSION = '0.1.0';

/**
 * Text model configuration extracted from manifest
 */
export interface TextModelConfig {
  numLayers: number;
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  numKVHeads: number;
  headDim: number;
  vocabSize: number;
  maxSeqLen: number;
  quantization: string;
}

/**
 * Inferred attention parameters from tensor shapes
 */
export interface InferredAttentionParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

/**
 * Model memory estimate
 */
export interface ModelEstimate {
  weightsBytes: number;
  kvCacheBytes: number;
  totalBytes: number;
  modelConfig: TextModelConfig;
}

/**
 * Progress callback event
 */
export interface LoadProgressEvent {
  stage: 'connecting' | 'manifest' | 'estimate' | 'warming' | 'downloading' | 'loading';
  message: string;
  estimate?: ModelEstimate;
}

/**
 * Generation options
 */
export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopTokens?: number[];
  stopSequences?: string[];
  onToken?: (token: string) => void;
}

/**
 * Chat message format
 */
export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

/**
 * Chat response format
 */
export interface ChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

/**
 * DOPPLER capability flags type
 */
export interface DopplerCapabilitiesType {
  available: boolean;
  HAS_MEMORY64: boolean;
  HAS_SUBGROUPS: boolean;
  HAS_F16: boolean;
  IS_UNIFIED_MEMORY: boolean;
  TIER_LEVEL: number;
  TIER_NAME: string;
  MAX_MODEL_SIZE: number;
  initialized: boolean;
  currentModelId: string | null;
  kernelsWarmed: boolean;
  kernelsTuned: boolean;
  lastModelEstimate: ModelEstimate | null;
  bridgeClient?: ExtensionBridgeClient | null;
  localPath?: string | null;
}

/**
 * Provider interface for llm-client.js integration
 */
export interface DopplerProviderInterface {
  name: string;
  displayName: string;
  isLocal: boolean;
  init(): Promise<boolean>;
  loadModel(
    modelId: string,
    modelUrl?: string | null,
    onProgress?: ((event: LoadProgressEvent) => void) | null,
    localPath?: string | null
  ): Promise<boolean>;
  chat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse>;
  stream(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string>;
  prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  generateWithPrefixKV(prefix: KVCacheSnapshot, prompt: string, options?: GenerateOptions): AsyncGenerator<string>;
  loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void>;
  unloadLoRAAdapter(): Promise<void>;
  getActiveLoRA(): string | null;
  getCapabilities(): DopplerCapabilitiesType;
  getModels(): Promise<string[]>;
  destroy(): Promise<void>;
}

/**
 * DOPPLER capability flags (populated at init)
 */
export const DopplerCapabilities: DopplerCapabilitiesType = {
  available: false,
  HAS_MEMORY64: false,
  HAS_SUBGROUPS: false,
  HAS_F16: false,
  IS_UNIFIED_MEMORY: false,
  TIER_LEVEL: 1,
  TIER_NAME: '',
  MAX_MODEL_SIZE: 0,
  initialized: false,
  currentModelId: null,
  kernelsWarmed: false,
  kernelsTuned: false,
  lastModelEstimate: null,
};
