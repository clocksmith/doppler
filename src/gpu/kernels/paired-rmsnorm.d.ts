import type { Tensor } from '../tensor.js';
import type { CommandRecorder } from '../command-recorder.js';
import type { WeightBuffer, TensorLike } from '../weight-buffer.js';
import type { RMSNormOptions } from './rmsnorm.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
/** Residual fusion and weight offset a paired RMSNorm call site requests. */
export interface PairedRMSNormFusion {
  rmsNormWeightOffset: boolean;
  preResidual: boolean;
  residual: boolean;
  outputPrenorm: boolean;
}
/**
 * The paired RMSNorm program to dispatch: the execution path's declared variant, or `paired`
 * when no path pins one. Fails closed unless the registry declares a paired program.
 */
export declare function resolvePairedRMSNormVariant(declaredVariant: string | null): string;
/**
 * Interim Q/K head-normalization program of a paired graph that declares no `q_norm`/`k_norm`
 * step (Capsules signed before the steps existed). Admission and Rig signing require such a
 * graph's closure to declare it. Removable once every supported manifest declares the steps.
 */
export declare const UNDECLARED_PAIRED_QK_NORM_VARIANT: 'paired';
/**
 * The paired program a graph's `q_norm` or `k_norm` step declares for a phase and layer; the
 * interim program when the graph declares neither step. Requires the execution graph's kernel
 * path, and fails closed unless the registry declares a paired program.
 */
export declare function resolvePairedQKNormVariant(op: 'q_norm' | 'k_norm', phase: 'prefill' | 'decode',
  layerIdx: number, kernelPath: import('../../config/schema/kernel-path.schema.js').KernelPathSchema | null | undefined): string;
/** Resolve the pipeline executePairedRMSNorm dispatches; it does not depend on token or hidden counts. */
export declare function resolvePairedRMSNormPipeline(weightDtype: string | null | undefined,
  fusion: PairedRMSNormFusion, variant: string): KernelPipelineRequest;
/** Registered paired RMSNorm execution, preserving source epsilon in two words. */
export declare function executePairedRMSNorm(recorder: CommandRecorder | null, input: Tensor,
  weight: GPUBuffer | WeightBuffer | TensorLike, eps: number,
  options: RMSNormOptions & { batchSize: number; hiddenSize: number; variant: string }): Promise<Tensor>;
