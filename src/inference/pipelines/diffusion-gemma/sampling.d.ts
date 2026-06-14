import type { DiffusionGemmaConfig } from './config.js';

export interface DiffusionGemmaLogitsRequest {
  canvas: Int32Array;
  step: number;
  temperature: number;
  selfConditioningLogits: ArrayBufferView | null;
  canvasIndex: number;
  inputIds: Int32Array | null;
}

export type DiffusionGemmaLogitsProvider = (
  request: DiffusionGemmaLogitsRequest
) => ArrayBufferView | Promise<ArrayBufferView>;

export interface DiffusionGemmaStatsRequest {
  canvas: Int32Array;
  step: number;
  temperature: number;
  selfConditioningLogits: unknown;
  canvasIndex: number;
  inputIds: Int32Array | null;
}

export interface DiffusionGemmaStatsProviderResult {
  argmaxCanvas: Int32Array;
  entropies: Float32Array;
  selfConditioningLogits?: unknown;
}

export type DiffusionGemmaStatsProvider = (
  request: DiffusionGemmaStatsRequest
) => DiffusionGemmaStatsProviderResult | Promise<DiffusionGemmaStatsProviderResult>;

export interface DiffusionGemmaStepResult {
  canvas: Int32Array;
  argmaxCanvas: Int32Array;
  sampledCanvas: Int32Array;
  entropies: Float32Array;
  processedLogits: Float32Array | null;
  accepted: Uint8Array;
  acceptedCount: number;
  meanEntropy: number;
}

export interface DiffusionGemmaStabilityState {
  counts: Uint16Array;
  stableTokens: number;
  allStable: boolean;
}

export interface DiffusionGemmaDenoiseResult {
  canvas: Int32Array;
  argmaxCanvas: Int32Array | null;
  selfConditioningLogits: ArrayBufferView | unknown | null;
  lastStep: (DiffusionGemmaStepResult & {
    step: number;
    temperature: number;
    stability: DiffusionGemmaStabilityState;
  }) | null;
  stepsRun: number;
}

export function createSeededRandom(seed: number): () => number;

export function resolveDenoisingTemperature(
  config: DiffusionGemmaConfig,
  step: number
): number;

export function initializeCanvas(
  config: DiffusionGemmaConfig,
  random: () => number
): Int32Array;

export function applyEntropyBoundStep(
  currentCanvas: ArrayLike<number>,
  logits: ArrayBufferView,
  config: DiffusionGemmaConfig,
  options: {
    random: () => number;
    temperature: number;
  }
): DiffusionGemmaStepResult;

export function applyEntropyBoundStatsStep(
  currentCanvas: ArrayLike<number>,
  stats: {
    argmaxCanvas: ArrayLike<number>;
    entropies: Float32Array;
  },
  config: DiffusionGemmaConfig,
  options: {
    random: () => number;
    temperature: number;
  }
): DiffusionGemmaStepResult;

export function updateStabilityState(
  previousArgmaxCanvas: ArrayLike<number> | null,
  argmaxCanvas: ArrayLike<number>,
  previousCounts: ArrayLike<number> | null,
  config: DiffusionGemmaConfig
): DiffusionGemmaStabilityState;

export function denoiseCanvas(
  config: DiffusionGemmaConfig,
  options: {
    logitsProvider: DiffusionGemmaLogitsProvider;
    random: () => number;
    initialCanvas?: ArrayLike<number> | null;
    selfConditioningLogits?: ArrayBufferView | null;
    canvasIndex?: number;
    inputIds?: Int32Array | null;
  }
): Promise<DiffusionGemmaDenoiseResult>;

export function denoiseCanvasWithStatsProvider(
  config: DiffusionGemmaConfig,
  options: {
    statsProvider: DiffusionGemmaStatsProvider;
    random: () => number;
    initialCanvas?: ArrayLike<number> | null;
    selfConditioningLogits?: unknown;
    canvasIndex?: number;
    inputIds?: Int32Array | null;
  }
): Promise<DiffusionGemmaDenoiseResult>;
