import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { KernelPathSchema } from '../../../../config/schema/index.js';
import type { HyperConnectionProjection } from './decoder-layer.js';

export declare function createHyperConnectionProjection(options: {
  recorder?: CommandRecorder | null;
  kernelPath: KernelPathSchema | null | undefined;
  phase: 'prefill' | 'decode' | null | undefined;
  layerIdx: number;
}): HyperConnectionProjection;

/** Explicit hc_mix_branch declaration; absent leaves the separate operators active. */
export declare function resolveHyperConnectionMixBranch(options: {
  kernelPath: KernelPathSchema | null | undefined;
  phase: 'prefill' | 'decode' | null | undefined;
  layerIdx: number;
}): boolean;
