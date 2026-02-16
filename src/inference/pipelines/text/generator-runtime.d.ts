import type { PipelineState } from './state.js';
import type { GenerateOptions } from './types.js';
import type { ParsedModelConfig } from './config.js';

export interface StepOptionsResolved {
  temperature: number;
  topP: number;
  topK: number;
  repetitionPenalty: number;
  debug: boolean;
  debugLayers?: number[];
  profile: boolean;
  disableCommandBatching: boolean;
  disableMultiTokenDecode: boolean;
  batchSize: number;
  stopCheckMode: 'batch' | 'per-token';
}

export interface GenerateOptionsResolved extends StepOptionsResolved {
  maxTokens: number;
  stopSequences: string[];
  useSpeculative: boolean;
  useChatTemplate: boolean;
  benchmark: boolean;
}

export interface PrefillOptionsResolved {
  useChatTemplate: boolean;
  debug: boolean;
  debugLayers?: number[];
  profile: boolean;
  disableCommandBatching: boolean;
  disableMultiTokenDecode: boolean;
}

export interface PrefillEmbeddingOptionsResolved extends PrefillOptionsResolved {
  embeddingMode: 'last' | 'mean';
}

export function assertTokenIdsInRange(state: PipelineState, tokenIds: number[], context?: string): void;
export function assertTokenIdInRange(state: PipelineState, tokenId: number, context?: string): void;

export function resolveStepOptions(
  state: PipelineState,
  options?: GenerateOptions
): StepOptionsResolved;

export function resolveGenerateOptions(
  state: PipelineState,
  options?: GenerateOptions
): GenerateOptionsResolved;

export function resolvePrefillOptions(
  state: PipelineState,
  options?: GenerateOptions
): PrefillOptionsResolved;

export function resolvePrefillEmbeddingOptions(
  state: PipelineState,
  options?: GenerateOptions
): PrefillEmbeddingOptionsResolved;

export function resolveAdvanceEmbeddingMode(
  state: PipelineState,
  options?: GenerateOptions
): 'last' | 'mean';

export function resolveFloatDtypeFromByteSize(
  totalBytes: number,
  expectedLength: number,
  fallback?: 'f16' | 'f32'
): 'f16' | 'f32';

export function decodeFloatWeights(
  data: ArrayBuffer,
  dtype: 'f16' | 'f32' | 'bf16',
  expectedLength: number,
  label: string
): Float32Array;

export function getFinalNormWeights(state: PipelineState): Promise<Float32Array>;

export function extractEmbeddingFromHidden(
  hiddenStates: Float32Array,
  numTokens: number,
  hiddenSize: number,
  embeddingMode: 'last' | 'mean',
  finalNormWeights: Float32Array,
  config: Pick<ParsedModelConfig, 'rmsNormEps' | 'rmsNormWeightOffset'>
): Float32Array;
