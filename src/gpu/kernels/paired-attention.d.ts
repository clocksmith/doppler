import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { AttentionOptions } from './attention.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
/**
 * A registry attention variant that declares paired Q/output and F16 K/V
 * (`paired` or `paired_tiled`); sequence and cache lengths are uniforms.
 */
export declare function resolvePairedAttentionPipeline(variant: string): KernelPipelineRequest;
/**
 * The paired attention variant the execution path selects for this phase and
 * layer; `paired` when no path pins one. Fails closed on a non-paired variant.
 */
export declare function resolvePairedAttentionVariant(
  kernelPath: import('../../config/schema/kernel-path.schema.js').KernelPathSchema | null,
  phase: 'prefill' | 'decode',
  layerIdx: number | undefined
): string;
/** Declared paired Q/output and ordinary F16 contiguous K/V online attention. */
export declare function executePairedAttention(recorder: CommandRecorder | null, Q: Tensor, K: Tensor, V: Tensor, mask: GPUBuffer | null, numHeads: number, headDim: number, options: AttentionOptions): Promise<Tensor>;
