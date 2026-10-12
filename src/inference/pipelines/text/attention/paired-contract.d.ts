import type { Tensor } from '../../../../gpu/tensor.js';
import type { LayerWeights } from '../weights.js';
export interface PairedAttentionAdmissionConfig {
  activationDtype?: string; inputDtype?: string; outputDtype?: string;
  normalizationType?: string; valueNorm?: boolean;
  diffusionGemmaDecoder?: boolean; sharedKVSourceLayerIdx?: number | null;
  storeSharedKV?: boolean; multimodalBidirectionalSpan?: unknown;
  precomputedInputNorm?: Tensor | null;
}
export declare function assertPairedAttentionExecution(input: Tensor,
  config: PairedAttentionAdmissionConfig,
  state: { skipKVCacheWrites?: boolean; kvCache?: { hasGPUCache?: () => boolean; kvDtype?: string; layout?: string } },
  session: { intermediatePrecision?: 'f32' | 'two-f32' }, weights: Partial<LayerWeights> | null): void;
