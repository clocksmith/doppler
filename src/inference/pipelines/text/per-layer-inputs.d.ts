import type { Tensor } from '../../../gpu/tensor.js';
import type { LayerContext } from './types.js';

export declare function preparePerLayerInputs(
  tokenIds: number[] | Uint32Array | GPUBuffer,
  inputEmbedsTensor: Tensor,
  context: LayerContext,
  options?: {
    numTokens?: number;
    indexOffset?: number;
  }
): Promise<(GPUBuffer | null)[] | null>;

export declare function createPerLayerInputTensor(
  buffer: GPUBuffer,
  numTokens: number,
  hiddenSizePerLayerInput: number,
  activationDtype: 'f16' | 'f32'
): Tensor;
