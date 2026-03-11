import type { Tensor } from '../../../../gpu/tensor.js';

export interface AttentionProjectionInputResult {
  oProjInput: Tensor;
  oProjInputTemp: Tensor | null;
}

export function prepareAttentionProjectionInput(
  attnForProjection: Tensor,
  matmulOutputDtype: string,
  castToF16: (tensor: Tensor) => Promise<Tensor>
): Promise<AttentionProjectionInputResult>;
