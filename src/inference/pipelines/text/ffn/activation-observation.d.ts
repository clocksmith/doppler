import type { Tensor } from '../../../../gpu/tensor.js';
import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
export interface FfnObservationContext {
  operatorDiagnostics?: { activity?: import('../../../../debug/activity-trace.js').ActivityTrace } | null;
  recorder?: CommandRecorder | null;
  debugLayers?: number[] | null;
  trainingCapture?: {
    layerIdx: number; stage: string;
    capture(value: { layerIdx: number; stage: string; tensor: Tensor; numTokens: number;
      hiddenSize: number; recorder: CommandRecorder | null }): Promise<void>;
  } | null;
}
export function captureTrainingActivation(context: FfnObservationContext, layerIdx: number,
  tensor: Tensor, numTokens: number, hiddenSize: number): Promise<void>;
export function enqueueRecordedDenseHealth(context: FfnObservationContext, layerIdx: number,
  label: string, tensor: Tensor, elementCount: number): void;

export function captureProjectionInput(context: { operatorDiagnostics?: { activity?: import('../../../../debug/activity-trace.js').ActivityTrace } | null;
  recorder?: import('../../../../gpu/command-recorder.js').CommandRecorder | null }, layerIdx: number,
  tensor: import('../../../../gpu/tensor.js').Tensor, numTokens: number, width: number): Promise<void>;
