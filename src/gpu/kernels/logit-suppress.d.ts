import type { CommandRecorder } from '../command-recorder.js';
export declare function recordSuppressLogits(recorder: CommandRecorder, logits: GPUBuffer,
  vocabSize: number, tokenIds: number[]): Promise<void>;
