import type { Tensor } from '../tensor.js';
import type { LinearAttentionCoreLayerState, RunLinearAttentionCoreGPUOptions } from './linear-attention-core.js';
export declare function executePairedLinearAttention(qkv: Tensor, z: Tensor, a: Tensor, b: Tensor, state: LinearAttentionCoreLayerState, options: RunLinearAttentionCoreGPUOptions): Promise<Tensor>;
