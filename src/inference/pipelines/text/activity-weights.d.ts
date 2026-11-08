import type { ActivityLimits } from '../../../debug/activity-trace.js';
import type { ActivityWeight } from '../../../debug/activity-trace.js';
export function captureActivityWeights(weights: Map<string, unknown>, limits: ActivityLimits,
  signal?: AbortSignal, tiedReference?: boolean): Promise<ActivityWeight[]>;

export function prepareActivityRecording(state: import('./state.js').PipelineState, options: {
  __activationTrace?: import('../../../debug/activation-trace.js').ActivationTrace; signal?: AbortSignal;
}): Promise<void>;
