import type { PipelineState } from "../../state.js";
import type { ParsedModelConfig } from "../../config.js";
import type { DiagnosticSequenceState } from "../session/diagnostic-sequence.js";
import type { GenerateOptions } from "../../types.js";
import type { PromptInput } from "../../pipeline.js";

export interface TargetTokenLogProbsResult {
  seqLen: number;
  /** Prompt token ids exactly as prefilled before the target continuation. */
  tokens: number[];
  targetTokenIds: number[];
  /** Raw logit of each target token at its predicting position (after Doppler's finalize/softcap). */
  logits: number[];
  /** Natural-log log-sum-exp of each complete vocabulary row. */
  logSumExp: number[];
  /** logits[i] - logSumExp[i]: natural-log probability of target token i given every earlier token. */
  logProbs: number[];
  phase: Record<string, unknown>;
}

export interface TargetScoringGenerator {
  _state: PipelineState & DiagnosticSequenceState & Record<string, unknown> & {
    modelConfig: ParsedModelConfig;
    stats: PipelineState['stats'] & { operatorDiagnostics?: unknown };
  };
  /** Clears KV, linear-attention, conv and hyper-connection sequence state and sets currentSeqLen to 0. */
  _resetReplayPrefillRuntimeState(): void;
  _resetDecodeRuntimeState(): void;
  _assertTokenIdsInRange(tokenIds: number[], label: string): void;
  _closeFinitenessFallbackWindow(options: unknown): void;
  _prefillPromptToLogits(prompt: PromptInput, opts: unknown, contextLabel: string): Promise<{
    inputIds: number[];
    logits: unknown;
    phase?: Record<string, unknown> | null;
  }>;
}

/**
 * Teacher-forced scoring of `targetTokenIds` as the continuation of `options.inputIds`.
 * Starts from a fresh sequence (state left by an earlier generation is cleared), runs one
 * prefill over prompt + target[0..T-2] and projects only the T trailing rows to
 * complete-vocabulary logits.
 */
export declare function prefillWithTargetTokenLogProbs(
  this: TargetScoringGenerator,
  prompt: PromptInput,
  targetTokenIds: readonly number[],
  options: GenerateOptions & {
    inputIds: number[];
    __internalGenerate?: boolean;
    diagnostics?: { enabled?: boolean; captureConfig?: Partial<import("../../../../../debug/capture-policy.js").CaptureConfig> };
  }
): Promise<TargetTokenLogProbsResult>;
