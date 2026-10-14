import type { Tensor } from '../tensor.js';
import type { WeightBuffer } from '../weight-buffer.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { MatmulOptions } from './matmul.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
/** The single paired matmul program selected by weight encoding, layout and output precision. */
export interface PairedMatmulPipeline extends KernelPipelineRequest {
  readonly constants: Readonly<{ WORKGROUP_SIZE: number; WEIGHT_F16: boolean; WEIGHT_Q4K: boolean;
    TRANSPOSE_B: boolean; SCALAR_OUTPUT: boolean }>;
}
/**
 * Resolve the pipeline executePairedMatmul dispatches for this weight; it does not depend on M, N or K.
 * The weight encoding and output form select a program by rule; `programVariant`, the execution
 * step's declared variant, may name a registry program that replaces exactly that one
 * (`variantMetadata.replacesMechanism`).
 */
export declare function resolvePairedMatmulPipeline(weight: GPUBuffer | WeightBuffer | Tensor,
  options: Pick<MatmulOptions, 'transposeB'> & { outputPrecision?: 'f32' | 'two-f32'; programVariant?: string | null }): PairedMatmulPipeline;
/** Paired reduction with unchanged materialized weight values. */
export declare function executePairedMatmul(recorder: CommandRecorder | null, input: Tensor,
  weight: GPUBuffer | WeightBuffer, M: number, N: number, K: number,
  options: MatmulOptions & { outputPrecision?: 'f32' | 'two-f32'; programVariant?: string | null }): Promise<Tensor>;
