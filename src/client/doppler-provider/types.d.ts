import type { ExtensionBridgeClient } from '../../bridge/index.js';
import type { InferencePipeline, KVCacheSnapshot } from '../../inference/pipeline.js';
import type { LoRAManifest } from '../../adapters/lora-loader.js';
import type { RDRRManifest } from '../../storage/rdrr-format.js';

export declare const DOPPLER_PROVIDER_VERSION: string;

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

export interface InferredAttentionParams {
  numHeads: number;
  numKVHeads: number;
  headDim: number;
}

export interface ModelEstimate {
  weightsBytes: number;
  kvCacheBytes: number;
  totalBytes: number;
  modelConfig: TextModelConfig;
}

export interface LoadProgressEvent {
  stage: 'connecting' | 'manifest' | 'estimate' | 'warming' | 'downloading' | 'loading';
  message: string;
  estimate?: ModelEstimate;
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopTokens?: number[];
  stopSequences?: string[];
  useChatTemplate?: boolean;
  onToken?: (token: string) => void;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  content: string;
  usage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

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
  unloadModel(): Promise<void>;
  chat(messages: ChatMessage[], options?: GenerateOptions): Promise<ChatResponse>;
  stream(messages: ChatMessage[], options?: GenerateOptions): AsyncGenerator<string>;
  prefillKV(prompt: string, options?: GenerateOptions): Promise<KVCacheSnapshot>;
  generateWithPrefixKV(prefix: KVCacheSnapshot, prompt: string, options?: GenerateOptions): AsyncGenerator<string>;
  loadLoRAAdapter(adapter: LoRAManifest | RDRRManifest | string): Promise<void>;
  unloadLoRAAdapter(): Promise<void>;
  getActiveLoRA(): string | null;
  getPipeline(): InferencePipeline | null;
  getCurrentModelId(): string | null;
  getCapabilities(): DopplerCapabilitiesType;
  getModels(): Promise<string[]>;
  getAvailableModels(): Promise<string[]>;
  destroy(): Promise<void>;
}

export declare const DopplerCapabilities: DopplerCapabilitiesType;
