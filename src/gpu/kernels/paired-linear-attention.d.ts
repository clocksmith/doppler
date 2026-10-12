import type { Tensor } from '../tensor.js';
import type { LinearAttentionCoreLayerState, RunLinearAttentionCoreGPUOptions } from './linear-attention-core.js';
/** Compile the same provider-scoped pipelines used by paired recurrent execution. An abort waits for any started preparation to settle. */
export declare function preparePairedLinearAttentionPipelines(device: GPUDevice | undefined, signal?: AbortSignal): Promise<void>;
export declare function executePairedLinearAttention(qkv: Tensor, z: Tensor, a: Tensor, b: Tensor, state: LinearAttentionCoreLayerState, options: RunLinearAttentionCoreGPUOptions): Promise<Tensor>;
