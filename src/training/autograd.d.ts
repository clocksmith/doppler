import type { Tensor } from '../gpu/tensor.js';
import type { BackwardRegistrySchema } from '../config/schema/backward-registry.schema.js';

export const OpType: {
  EMBED: 'embed';
  MATMUL: 'matmul';
  RMSNORM: 'rmsnorm';
  ATTENTION: 'attention';
  SOFTMAX: 'softmax';
  ROPE: 'rope';
  SILU: 'silu';
  GELU: 'gelu';
  SCALE: 'scale';
  CROSS_ENTROPY: 'cross_entropy';
};

export interface AutogradRecord {
  op: string;
  inputs: Tensor[];
  output: Tensor;
  options?: Record<string, unknown>;
}

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
  backward(gradOutput: Tensor): Promise<Map<Tensor, Tensor>>;
  reset(): void;
}
