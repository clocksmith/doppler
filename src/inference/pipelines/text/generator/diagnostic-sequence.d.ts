import type { CaptureConfig } from '../../../../debug/capture-policy.js';
import type { OperatorEventEmitter, OperatorExecutionRecord } from '../operator-events.js';
import type { createTsirFixtureState } from '../tsir-fixture-writer.js';

export interface OperatorDiagnosticsSession {
  enabled: boolean;
  captureConfig: CaptureConfig;
  emitter: OperatorEventEmitter | null;
  tsirFixture: unknown;
}

export interface DiagnosticSequenceState {
  isGenerating: boolean;
  currentSeqLen: number;
  manifest?: { modelId?: string | null } | null;
  resolvedKernelPath?: { id?: string | null } | null;
  operatorDiagnostics: OperatorDiagnosticsSession | null;
  diagnosticSequence: { steps: number } | null;
}

export interface DiagnosticSequenceResult {
  enabled: boolean;
  /** Calls recorded in the sequence (prefill and decode steps). */
  steps: number;
  timeline: OperatorExecutionRecord[];
  recordCount: number;
}

/** Open one operator-diagnostics session for a forced-prefix sequence. */
export declare function beginDiagnosticSequence(
  state: DiagnosticSequenceState,
  options?: { captureConfig?: Partial<CaptureConfig> }
): void;

/**
 * Label the next records with the call's phase, absolute position (decode) and sequence
 * step. False when no sequence is open; throws for per-call diagnostics inside one.
 */
export declare function markDiagnosticSequenceStep(
  state: DiagnosticSequenceState,
  phase: 'prefill' | 'decode',
  options?: { diagnostics?: unknown } | null
): boolean;

/** Close the sequence and return its timeline. */
export declare function endDiagnosticSequence(state: DiagnosticSequenceState): DiagnosticSequenceResult;

/**
 * One prefill-family call's own diagnostics session when no sequence is open: enabled by
 * options.diagnostics or the diagnose harness mode, or for a TSIR fixture.
 */
export declare function beginPrefillDiagnostics(
  state: DiagnosticSequenceState & { stats: { operatorDiagnostics?: unknown }; runtimeConfig?: { shared?: { harness?: { mode?: string } } } },
  options: { diagnostics?: { enabled?: boolean; captureConfig?: Partial<CaptureConfig> } } | null | undefined,
  executionPlan: { planId?: string | null } | null | undefined,
  tsirFixtureCfg?: Parameters<typeof createTsirFixtureState>[0]
): void;

/** Close the call's finiteness window and diagnostics session, recording its timeline in stats. */
export declare function closePrefillDiagnostics(
  pipeline: { _state: DiagnosticSequenceState & { stats: { operatorDiagnostics?: unknown } }; _closeFinitenessFallbackWindow(options: unknown): void },
  options: unknown
): void;
