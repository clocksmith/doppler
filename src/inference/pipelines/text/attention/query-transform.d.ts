import type { CommandRecorder } from '../../../../gpu/kernel-selector.js';
import type { Tensor } from '../../../../gpu/tensor.js';

export function applyAttentionQueryScale(options: {
  recorder: CommandRecorder;
  tensor: Tensor;
  scale: unknown;
  count: number;
  release(buffer: GPUBuffer): void;
  observe?(tensor: Tensor): void | Promise<void>;
}): Promise<Tensor>;
