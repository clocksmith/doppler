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
/** Resolve the pipeline executePairedRMSNorm dispatches; it does not depend on token or hidden counts. */
export declare function resolvePairedRMSNormPipeline(weightDtype: string | null | undefined,
  fusion: PairedRMSNormFusion): KernelPipelineRequest;
/** Registered paired RMSNorm execution, preserving source epsilon in two words. */
export declare function executePairedRMSNorm(recorder: CommandRecorder | null, input: Tensor,
  weight: GPUBuffer | WeightBuffer | TensorLike, eps: number,
  options: RMSNormOptions & { batchSize: number; hiddenSize: number }): Promise<Tensor>;
