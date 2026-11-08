import type { ActivityTrace } from '../../../../debug/activity-trace.js';
import type { Tensor } from '../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { LinearLayerRuntimeState } from '../linear-attention.js';
export function observeLinearActivity(activity: ActivityTrace | null | undefined, layerIdx: number,
  state: LinearLayerRuntimeState, layout: { numVHeads: number }, a: Tensor, b: Tensor,
  numTokens: number, bOffset: number, recorder: CommandRecorder | null | undefined): Promise<void>;
