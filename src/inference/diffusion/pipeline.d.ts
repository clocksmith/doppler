/**
 * Diffusion Pipeline (scaffold)
 *
 * @module inference/diffusion/pipeline
 */

import type { DiffusionRequest, DiffusionResult, DiffusionStats, DiffusionRuntimeConfig } from './types.js';

export declare class DiffusionPipeline {
  runtimeConfig: { inference?: { diffusion?: DiffusionRuntimeConfig } } | null;
  manifest: Record<string, unknown> | null;
  diffusionState: Record<string, unknown> | null;
  tokenizers: Record<string, unknown> | null;
  stats: DiffusionStats;

  initialize(contexts?: Record<string, unknown>): Promise<void>;
  loadModel(manifest: Record<string, unknown>): Promise<void>;
  getStats(): DiffusionStats;
  getMemoryStats(): { used: number; kvCache: null };
  unload(): Promise<void>;
  generate(request: DiffusionRequest): Promise<DiffusionResult>;
}

export declare function createDiffusionPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<DiffusionPipeline>;
