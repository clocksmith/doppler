import type { PipelineState } from "../../state.js";
import type { ParsedModelConfig } from "../../config.js";
import type { GenerateOptions } from "../../types.js";
import type { CommandRecorder } from "../../../../../gpu/command-recorder.js";
import type { GpuLogitsResult } from "../sampling/token-selection.js";

export function computePrefillLogits(context: {
  config: ParsedModelConfig;
  state: PipelineState;
  currentRecorder: CommandRecorder | null | undefined;
  opts: GenerateOptions & { _returnGpuLogits?: boolean; _returnAllLogits?: boolean };
  selectedLogitTokenIds: readonly number[] | null;
  numTokens: number;
  tracePrefillEnabled: boolean;
  prefillBuffers: { hidden: GPUBuffer };
  recordProfile(recorder: CommandRecorder): Promise<void>;
  debugCheckBuffer: Parameters<typeof import("../../logits/index.js").computeLogits>[7];
  preserveBufferAcrossRecorderSubmit(buffer: GPUBuffer, recorder: CommandRecorder, label: string): GPUBuffer;
  submitFinalRecorder(recorder: CommandRecorder): Promise<void>;
}): Promise<Float32Array | GpuLogitsResult>;
