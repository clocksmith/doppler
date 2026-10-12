import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { AttentionOptions } from './attention.js';
/** Declared paired Q/output and ordinary F16 contiguous K/V online attention. */
export declare function executePairedAttention(recorder: CommandRecorder | null, Q: Tensor, K: Tensor, V: Tensor, mask: GPUBuffer | null, numHeads: number, headDim: number, options: AttentionOptions): Promise<Tensor>;
