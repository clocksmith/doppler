import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';

export interface EnergyEvalOptions {
  count?: number;
  scale?: number;
  outputBuffer?: GPUBuffer | null;
}

export interface EnergyUpdateOptions {
  count?: number;
  stepSize?: number;
  gradientScale?: number;
}

export interface EnergyQuintelUpdateOptions {
  count?: number;
  size?: number;
  flags?: number;
  stepSize?: number;
  gradientScale?: number;
  countDiff?: number;
  symmetryWeight?: number;
  countWeight?: number;
  centerWeight?: number;
  binarizeWeight?: number;
  centerTarget?: number;
  clampMin?: number;
  clampMax?: number;
}

export interface EnergyQuintelReduceOptions {
  count?: number;
  size?: number;
  flags?: number;
  symmetryWeight?: number;
  centerWeight?: number;
  binarizeWeight?: number;
  centerTarget?: number;
  outputBuffer?: GPUBuffer | null;
}

export interface EnergyQuintelGradOptions {
  count?: number;
  size?: number;
  flags?: number;
  countDiff?: number;
  symmetryWeight?: number;
  countWeight?: number;
  centerWeight?: number;
  binarizeWeight?: number;
  centerTarget?: number;
  outputBuffer?: GPUBuffer | null;
}

export declare function runEnergyEval(
  state: Tensor,
  target: Tensor,
  options?: EnergyEvalOptions
): Promise<Tensor>;

export declare function recordEnergyEval(
  recorder: CommandRecorder,
  state: Tensor,
  target: Tensor,
  options?: EnergyEvalOptions
): Promise<Tensor>;

export declare function runEnergyUpdate(
  state: Tensor,
  target: Tensor,
  options?: EnergyUpdateOptions
): Promise<Tensor>;

export declare function recordEnergyUpdate(
  recorder: CommandRecorder,
  state: Tensor,
  target: Tensor,
  options?: EnergyUpdateOptions
): Promise<Tensor>;

export declare function runEnergyQuintelUpdate(
  state: Tensor,
  options?: EnergyQuintelUpdateOptions
): Promise<Tensor>;

export declare function recordEnergyQuintelUpdate(
  recorder: CommandRecorder,
  state: Tensor,
  options?: EnergyQuintelUpdateOptions
): Promise<Tensor>;

export declare function runEnergyQuintelReduce(
  state: Tensor,
  options?: EnergyQuintelReduceOptions
): Promise<Tensor>;

export declare function runEnergyQuintelGrad(
  state: Tensor,
  options?: EnergyQuintelGradOptions
): Promise<Tensor>;

export declare function recordEnergyQuintelReduce(
  recorder: CommandRecorder,
  state: Tensor,
  options?: EnergyQuintelReduceOptions
): Promise<Tensor>;

export declare function recordEnergyQuintelGrad(
  recorder: CommandRecorder,
  state: Tensor,
  options?: EnergyQuintelGradOptions
): Promise<Tensor>;
