import type { Tensor } from '../tensor.js';

export interface Gemma4RouteQ4MatmulOptions {
  numRoutes: number;
  topK: number;
  N: number;
  K: number;
  inputMode?: 'token' | 'route';
  alpha?: number;
  outputBuffer?: GPUBuffer | null;
  label?: string;
}

export declare function runGemma4RouteQ4MatmulF16A(
  input: Tensor,
  routeIndices: GPUBuffer,
  weight: unknown,
  options: Gemma4RouteQ4MatmulOptions
): Promise<Tensor>;

export interface ScatterAddRoutesF16ExpertScaleOptions {
  outputBuffer?: GPUBuffer | null;
  label?: string;
}

export declare function runScatterAddRoutesF16ExpertScale(
  routeOutputs: Tensor,
  routingIndices: GPUBuffer,
  routingWeights: GPUBuffer,
  expertScales: GPUBuffer,
  numTokens: number,
  hiddenSize: number,
  topK: number,
  options?: ScatterAddRoutesF16ExpertScaleOptions
): Promise<Tensor>;
