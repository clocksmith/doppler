import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
export interface SharedSlotGeometry {
  numTokens: number;
  numExperts: number;
  topK: number;
  hiddenSize: number;
  intermediateSize: number;
  sharedSize: number;
  swigluLimit: number | null;
}
export interface SharedSlotOptions extends SharedSlotGeometry {
  indices: GPUBuffer;
  /** Registered variant declared by the kernel-path step. */
  variant: 'native_f32' | 'pair_f32' | 'pair_combine_f32' | 'pair_combine_residual_f32';
  outputBuffer?: GPUBuffer | null;
  /** Pair down only: per-slot outputs [numTokens * (topK + 1), hiddenSize] before the fixed-order combine. */
  slotsBuffer?: GPUBuffer | null;
}
export declare function validateSharedSlotGeometry(options: SharedSlotGeometry): void;
export declare function executeSharedSlotGateUp(recorder: CommandRecorder | null, input: Tensor,
  weights: { gate: WeightBuffer; up: WeightBuffer; sharedGate: WeightBuffer; sharedUp: WeightBuffer },
  options: SharedSlotOptions): Promise<Tensor>;
/**
 * Returns the routed plus gated shared result. pair_combine_residual_f32 requires
 * `residual` [numTokens, hiddenSize] F32 and returns that sum plus the residual; every
 * other variant rejects `residual` and the layer owns the outer residual.
 */
export declare function executeSharedSlotDown(recorder: CommandRecorder | null, input: Tensor,
  weights: { down: WeightBuffer; sharedDown: WeightBuffer }, gate: Tensor,
  options: SharedSlotOptions & { routeWeights: GPUBuffer; residual?: Tensor | null }): Promise<Tensor>;
