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

/** Role-scoped decode fusion; original up role must remain explicitly Q4_0 gemv32. */
export declare function resolveHyperConnectionUpMixBranch(options: {
  kernelPath: KernelPathSchema | null | undefined;
  phase: 'prefill' | 'decode' | null | undefined;
  layerIdx: number;
  role: 'attn_hc' | 'mlp_hc';
}): boolean;
