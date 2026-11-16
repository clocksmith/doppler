import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
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

export interface RecordedAttentionObservation {
  recorder: CommandRecorder;
  config: Pick<import('./types.js').AttentionConfig, 'layerIdx' | 'numTokens' | 'numHeads' | 'numKVHeads' | 'headDim'>;
  state: import('./types.js').AttentionState;
  debugFlags: import('./types.js').AttentionDebugFlags;
  qTensor: Tensor;
  kTensor: Tensor | null;
}
export function observeRecordedAttentionProjections(context: RecordedAttentionObservation & {
  vTensor: Tensor | null;
}): Promise<void>;
export function observeRecordedAttentionQKNorm(context: RecordedAttentionObservation): Promise<void>;
