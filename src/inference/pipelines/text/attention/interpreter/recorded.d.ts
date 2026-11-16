export declare const resolveDirectF16KVCacheWrite: any;
export declare const runRecordedAttentionCore: any;
export declare const finalizeRecordedAttentionSuccess: any;

export function failRecordedAttention(context: {
  resourceScope: import('../../../../resource-scope.js').ResourceScope<GPUBuffer>;
  attnForProjection: import('../../../../../gpu/tensor.js').Tensor | null;
  attnOutput: import('../../../../../gpu/tensor.js').Tensor | null;
  qGateTensor: import('../../../../../gpu/tensor.js').Tensor | null;
  qTensor: import('../../../../../gpu/tensor.js').Tensor | null;
  kTensor: import('../../../../../gpu/tensor.js').Tensor | null;
  retainSharedKvBuffers: boolean;
  vTensor: import('../../../../../gpu/tensor.js').Tensor | null;
  normed: import('../../../../../gpu/tensor.js').Tensor | null;
  attentionInput: import('../../../../../gpu/tensor.js').Tensor | null;
  attentionInputTemp: boolean;
  decoderKTemp: import('../../../../../gpu/tensor.js').Tensor | null;
  decoderVTemp: import('../../../../../gpu/tensor.js').Tensor | null;
  decoderKVState: { ownedBuffers?: GPUBuffer[] | null } | null;
  state: import('../types.js').AttentionState;
  attentionPlan: import('../plan.js').SemanticAttentionPlan | null;
  error: unknown;
}): never;
