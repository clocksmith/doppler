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
  outputBuffer?: GPUBuffer | null;
}
export declare function validateSharedSlotGeometry(options: SharedSlotGeometry): void;
export declare function executeSharedSlotGateUp(recorder: CommandRecorder | null, input: Tensor,
  weights: { gate: WeightBuffer; up: WeightBuffer; sharedGate: WeightBuffer; sharedUp: WeightBuffer },
  options: SharedSlotOptions): Promise<Tensor>;
/** Returns the routed plus gated shared result. The layer owns the outer residual. */
export declare function executeSharedSlotDown(recorder: CommandRecorder | null, input: Tensor,
  weights: { down: WeightBuffer; sharedDown: WeightBuffer }, gate: Tensor,
  options: SharedSlotOptions & { routeWeights: GPUBuffer }): Promise<Tensor>;
