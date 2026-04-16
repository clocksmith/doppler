import type { Tensor, TensorDType } from '../../gpu/tensor.js';

export declare function createUploadedTensor(
  data: ArrayBufferView,
  dtype: TensorDType,
  shape: number[],
  label: string,
  usage?: number | undefined
): Tensor;
