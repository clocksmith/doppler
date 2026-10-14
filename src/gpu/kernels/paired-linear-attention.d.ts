import type { Tensor } from '../tensor.js';
import type { LinearAttentionCoreLayerState, RunLinearAttentionCoreGPUOptions } from './linear-attention-core.js';
import type { LinearNormGateActivation } from '../../config/schema/inference.schema.js';
import type { KernelPipelineRequest } from './pipeline-cache.js';
/**
 * The nine pipelines executePairedLinearAttention dispatches, in dispatch order.
 * The gate phase follows the model's norm-gate activation; no member depends on token count.
 */
export declare function listPairedLinearAttentionPipelines(
  normGateActivation: LinearNormGateActivation
): KernelPipelineRequest[];
/**
 * Compile the same provider-scoped pipelines used by paired recurrent execution,
 * including the gate phase selected for the model's norm-gate activation.
 * An abort waits for any started preparation to settle.
 */
export declare function preparePairedLinearAttentionPipelines(
  device: GPUDevice | undefined,
  signal: AbortSignal | undefined,
  normGateActivation: LinearNormGateActivation
): Promise<void>;
export declare function executePairedLinearAttention(qkv: Tensor, z: Tensor, a: Tensor, b: Tensor, state: LinearAttentionCoreLayerState, options: RunLinearAttentionCoreGPUOptions): Promise<Tensor>;
