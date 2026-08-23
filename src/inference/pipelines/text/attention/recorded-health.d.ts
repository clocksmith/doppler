import type { CommandRecorder } from '../../../../gpu/kernel-selector.js';
import type { Tensor } from '../../../../gpu/tensor.js';

export function shouldTraceRecordedHealth(
  layerIdx: number,
  debugFlags: { debugLayers?: number[] | null } | null | undefined
): boolean;
export function enqueueRecordedTensorHealth(
  recorder: CommandRecorder,
  label: string,
  tensor: Tensor,
  dtype: string,
  elementCount: number
): void;
