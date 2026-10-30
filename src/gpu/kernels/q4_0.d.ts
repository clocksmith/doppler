import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { MatmulOptions } from './matmul.js';
export interface Q40MatmulOptions extends MatmulOptions { variant?: string | null; signal?: AbortSignal | null }
export declare function q40Dispatch(device: GPUDevice | null, count: number): [number,number,number];
export declare function assertQ40Buffer(device: GPUDevice, buffer: GPUBuffer, bytes: number, usage: number, label: string): void;
export declare function runQ40Matmul(input: Tensor, weights: WeightBuffer, M: number, N: number, K: number, options: Q40MatmulOptions): Promise<Tensor>;
export declare function recordQ40Matmul(recorder: CommandRecorder, input: Tensor, weights: WeightBuffer, M: number, N: number, K: number, options: Q40MatmulOptions): Promise<Tensor>;
import type { GatherOptions } from './gather.js';
export declare function executeQ40Gather(recorder: CommandRecorder | null, indices: GPUBuffer, weights: GPUBuffer,
  numTokens: number, hiddenSize: number, vocabSize: number, options: GatherOptions): Promise<Tensor>;

export type Q40Execute = (recorder: CommandRecorder | null, input: Tensor, weights: WeightBuffer, M: number, N: number, K: number, options: Q40MatmulOptions) => Promise<Tensor>;
