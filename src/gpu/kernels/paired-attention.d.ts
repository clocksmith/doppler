import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { AttentionOptions } from './attention.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
/** The single declared paired attention program; sequence and cache lengths are uniforms. */
export declare function resolvePairedAttentionPipeline(): KernelPipelineRequest;
/** Declared paired Q/output and ordinary F16 contiguous K/V online attention. */
export declare function executePairedAttention(recorder: CommandRecorder | null, Q: Tensor, K: Tensor, V: Tensor, mask: GPUBuffer | null, numHeads: number, headDim: number, options: AttentionOptions): Promise<Tensor>;
