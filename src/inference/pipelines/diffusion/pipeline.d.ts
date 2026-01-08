/**
 * Diffusion Pipeline (scaffold)
 *
 * @module inference/pipelines/diffusion/pipeline
 */

import type { DiffusionRequest, DiffusionResult, DiffusionStats, DiffusionRuntimeConfig } from './types.js';
import type { DiffusionWeightLoader, DiffusionWeightEntry } from './weights.js';

export declare class DiffusionPipeline {
  runtimeConfig: { inference?: { diffusion?: DiffusionRuntimeConfig } } | null;
  manifest: Record<string, unknown> | null;
  diffusionState: Record<string, unknown> | null;
  tokenizers: Record<string, unknown> | null;
  stats: DiffusionStats;
  gpuScaffold: Record<string, unknown> | null;
  weightLoader: DiffusionWeightLoader | null;
  vaeWeights: DiffusionWeightEntry | null;
  textEncoderWeights: Record<string, DiffusionWeightEntry> | null;
  transformerWeights: DiffusionWeightEntry | null;

  initialize(contexts?: Record<string, unknown>): Promise<void>;
  loadModel(manifest: Record<string, unknown>): Promise<void>;
  getStats(): DiffusionStats;
  getMemoryStats(): { used: number; kvCache: null };
  unload(): Promise<void>;
  ensureTextEncoderWeights(): Promise<Record<string, DiffusionWeightEntry>>;
  ensureTransformerWeights(): Promise<DiffusionWeightEntry>;
  releaseTextEncoderWeights(): void;
  releaseTransformerWeights(): void;
  generate(request: DiffusionRequest): Promise<DiffusionResult>;
  generateCPU(request: DiffusionRequest): Promise<DiffusionResult>;
  generateGPU(request: DiffusionRequest): Promise<DiffusionResult>;
}

export declare function createDiffusionPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<DiffusionPipeline>;
