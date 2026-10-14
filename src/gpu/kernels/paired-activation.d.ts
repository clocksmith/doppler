import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { OutputBufferOptions } from './types.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
export interface PairedPointwiseOptions extends OutputBufferOptions { scale?: number; inplace?: boolean; }
export type PairedPointwiseOperation = 'add' | 'scale' | 'multiply' | 'silu' | 'sigmoid' | 'silu_unary';
/** A gated activation names its input activation; ungated operations use null. */
export interface PairedPointwiseMode { inplace: boolean; gatedInputActivation: 'silu' | 'identity' | null; }
/** Resolve the pipeline a paired pointwise call dispatches; it does not depend on the element count. */
export function resolvePairedPointwisePipeline(operation: PairedPointwiseOperation, mode: PairedPointwiseMode): KernelPipelineRequest;
export function executePairedAdd(target: CommandRecorder | null, a: Tensor, b: Tensor, count: number, options?: PairedPointwiseOptions): Promise<Tensor>;
export function executePairedScale(target: CommandRecorder | null, input: Tensor, scale: number, count: number, options?: PairedPointwiseOptions): Promise<Tensor>;
export function executePairedMultiply(target: CommandRecorder | null, a: Tensor, b: Tensor, count: number, options?: OutputBufferOptions): Promise<Tensor>;
export function executePairedSiLU(target: CommandRecorder | null, input: Tensor, count: number, options: OutputBufferOptions & { gate?: Tensor | null; gateActivation: 'silu' | 'sigmoid'; inputActivation: 'silu' | 'identity'; swigluLimit: number | null }): Promise<Tensor>;
