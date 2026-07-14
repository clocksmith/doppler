import type { Tensor } from '../../gpu/tensor.js';
import type { BackwardRegistrySchema } from '../../config/schema/backward-registry.schema.js';

export const OpType: {
  EMBED: 'embed';
  MATMUL: 'matmul';
  RMSNORM: 'rmsnorm';
  RESIDUAL_ADD: 'residual_add';
  RESHAPE: 'reshape';
  ROW_SLICE: 'row_slice';
  LAYERNORM: 'layernorm';
  ATTENTION: 'attention';
  SOFTMAX: 'softmax';
  ROPE: 'rope';
  SILU: 'silu';
  SILU_ROWSPLIT: 'silu_rowsplit';
  SILU_GATED: 'silu_gated';
  GELU: 'gelu';
  SCALE: 'scale';
  CROSS_ENTROPY: 'cross_entropy';
  BIAS_ADD: 'bias_add';
  UPSAMPLE2D: 'upsample2d';
  PIXEL_SHUFFLE: 'pixel_shuffle';
  GROUPNORM: 'groupnorm';
  CONV2D: 'conv2d';
};

export declare function resolveMatmulBackwardOptions<T extends Record<string, unknown> = Record<string, unknown>>(
  options?: T
): T & {
  computeGradInput: boolean;
  computeGradWeight: boolean;
};

export declare function computeSiluGatedBackwardValues(
  gate: ArrayLike<number>,
  up: ArrayLike<number>,
  gradOutput: ArrayLike<number>,
  swigluLimit?: number
): {
  gradGate: Float32Array;
  gradUp: Float32Array;
};

export interface AutogradRecord {
  op: string;
  inputs: Tensor[];
  output: Tensor;
  options?: Record<string, unknown>;
}

export interface BackwardSeed {
  tensor: Tensor;
  grad: Tensor;
}

export type BackwardSeedInput =
  | Tensor
  | Map<Tensor, Tensor>
  | BackwardSeed[]
  | { seeds: BackwardSeed[] };

export declare class AutogradTape {
  constructor(registry: BackwardRegistrySchema);
  registry: BackwardRegistrySchema;
  records: AutogradRecord[];
  watch<T extends Tensor>(tensor: T): T;
  record(
    op: string,
    fn: (...args: Tensor[]) => Promise<Tensor>,
    inputs: Tensor[],
    options?: Record<string, unknown>
  ): Promise<Tensor>;
  backward(gradOutput: BackwardSeedInput): Promise<Map<Tensor, Tensor>>;
  reset(): void;
}
