import type { Tensor } from '../tensor.js';
import type { KernelRecorder } from './types.js';

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
  rules?: {
    mirrorX?: boolean;
    mirrorY?: boolean;
    diagonal?: boolean;
    count?: boolean;
    center?: boolean;
  };
}

export declare function runEnergyEval(
  state: Tensor,
  target: Tensor,
  options?: EnergyEvalOptions
): Promise<Tensor>;

export declare function recordEnergyEval(
  recorder: KernelRecorder,
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
  recorder: KernelRecorder,
  state: Tensor,
  target: Tensor,
  options?: EnergyUpdateOptions
): Promise<Tensor>;

export declare function runEnergyQuintelUpdate(
  state: Tensor,
  options?: EnergyQuintelUpdateOptions
): Promise<Tensor>;

export declare function recordEnergyQuintelUpdate(
  recorder: KernelRecorder,
  state: Tensor,
  options?: EnergyQuintelUpdateOptions
): Promise<Tensor>;
