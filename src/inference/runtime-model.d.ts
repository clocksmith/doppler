import type {
  AdapterConfigSchema,
  KernelPathRef,
  ManifestInferenceSchema,
  MoEConfigSchema,
  ProvenanceSchema,
  QuantizationInfoSchema,
  WeightLayout,
} from '../config/schema/index.js';
import type { ComponentGroup, ConversionInfo, LayerConfig, ShardInfo, TensorMap } from '../formats/rdrr/index.js';

export interface RuntimeModelContract {
  kind: 'runtime-model';
  sourceFormat: string | null;
  version?: number;
  modelId: string;
  modelType: string;
  quantization: string;
  quantizationInfo?: QuantizationInfoSchema;
  hashAlgorithm: string;
  eos_token_id: number | number[] | null;
  image_token_id?: number;
  audio_token_id?: number;
  video_token_id?: number;
  architecture: LayerConfig | string | Record<string, unknown>;
  groups?: Record<string, ComponentGroup>;
  shards: ShardInfo[];
  totalSize: number;
  tensorsFile?: string;
  tensorCount?: number;
  tokenizer?: {
    type: string;
    file?: string;
    vocabSize: number;
    sentencepieceModel?: string;
    eosTokenId?: number;
    eosTokens?: number[];
    bosTokenId?: number;
    padTokenId?: number;
    unkTokenId?: number;
    addBosToken?: boolean;
    addEosToken?: boolean;
  };
  moeConfig?: MoEConfigSchema & { expertSize?: number };
  optimizations?: {
    kernelPath?: KernelPathRef;
  };
  runtime?: {
    useBatching?: boolean;
    debug?: boolean;
  };
  config?: Record<string, unknown>;
  quantization_config?: Record<string, unknown> | null;
  conversion?: ConversionInfo;
  inference: ManifestInferenceSchema;
  blake3Full?: string;
  defaultWeightLayout?: WeightLayout;
  metadata?: Record<string, unknown>;
  adapterType?: 'lora' | 'qlora';
  baseCompatibility?: string[];
  mergedAdapter?: AdapterConfigSchema;
  adapterConfig?: AdapterConfigSchema;
  provenance?: ProvenanceSchema;
  baseModel?: string;
  loraConfig?: {
    rank: number;
    alpha: number;
    targetModules?: string[];
    dropout?: number;
  };
  draftModel?: { numTokens?: number };
  tensors?: TensorMap;
}

export interface CreateRuntimeModelContractOptions
  extends Omit<RuntimeModelContract, 'kind' | 'sourceFormat'> {
  sourceFormat?: string | null;
}

export declare function createRuntimeModelContract(
  options: CreateRuntimeModelContractOptions
): RuntimeModelContract;

export declare function createRuntimeModelFromManifest<T extends Record<string, unknown>>(
  manifest: T,
  options?: { sourceFormat?: string | null }
): T & RuntimeModelContract;

export declare function isRuntimeModelContract(value: unknown): value is RuntimeModelContract;
