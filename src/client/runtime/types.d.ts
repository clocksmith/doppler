import type { InferencePipeline, KVCacheSnapshot } from '../../inference/pipelines/text.js';
import type { RDRRManifest } from '../../formats/rdrr/index.js';

export interface ExtensionBridgeClient {
  read(path: string, offset?: number, length?: number): Promise<ArrayBuffer | Uint8Array>;
  disconnect?(): void;
}

export interface LoRAManifest {
  adapterType?: string;
  modelType?: string;
  name?: string;
  rank?: number;
  tensors?: unknown;
  [key: string]: unknown;
}

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
  content: string | Array<Record<string, unknown>>;
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

export declare const DopplerCapabilities: DopplerCapabilitiesType;
