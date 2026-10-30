import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';

export interface RouteQ4MatmulOptions {
  numRoutes: number;
  numExperts: number;
  topK: number;
  N: number;
  K: number;
  inputMode: 'token' | 'route';
  alpha: number;
  outputBuffer?: GPUBuffer | null;
}
export interface RouteScatterOptions {
  numTokens: number;
  hiddenSize: number;
  topK: number;
  outputBuffer?: GPUBuffer | null;
}
export declare function runRouteQ4Matmul(input: Tensor, indices: GPUBuffer,
  weights: WeightBuffer, options: RouteQ4MatmulOptions): Promise<Tensor>;
export declare function recordRouteQ4Matmul(recorder: CommandRecorder, input: Tensor,
  indices: GPUBuffer, weights: WeightBuffer, options: RouteQ4MatmulOptions): Promise<Tensor>;
export declare function runRouteScatter(input: Tensor, weights: GPUBuffer,
  options: RouteScatterOptions): Promise<Tensor>;
export declare function recordRouteScatter(recorder: CommandRecorder, input: Tensor,
  weights: GPUBuffer, options: RouteScatterOptions): Promise<Tensor>;
