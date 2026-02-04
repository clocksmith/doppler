/**
 * Energy Pipeline
 *
 * @module inference/energy/pipeline
 */

import type { EnergyRequest, EnergyResult, EnergyStats } from './types.js';

export declare class EnergyPipeline {
  runtimeConfig: Record<string, unknown> | null;
  manifest: Record<string, unknown> | null;
  stats: EnergyStats;
  baseUrl: string | null;
  _onProgress: ((progress: { stage?: string; percent: number; message?: string }) => void) | null;

  initialize(contexts?: Record<string, unknown>): Promise<void>;
  loadModel(manifest: Record<string, unknown>): Promise<void>;
  getStats(): EnergyStats;
  getMemoryStats(): { used: number; kvCache: null };
  unload(): Promise<void>;
  generate(request?: EnergyRequest): Promise<EnergyResult>;
}

export declare function createEnergyPipeline(
  manifest: Record<string, unknown>,
  contexts?: Record<string, unknown>
): Promise<EnergyPipeline>;
