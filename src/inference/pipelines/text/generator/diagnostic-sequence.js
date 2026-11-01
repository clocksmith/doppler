import { CAPTURE_LEVELS, createDefaultCaptureConfig, validateCaptureConfig } from '../../../../debug/capture-policy.js';
import { OperatorEventEmitter } from '../operator-events.js';
import { createTsirFixtureState } from '../tsir-fixture-writer.js';

// One operator-diagnostics session across a forced-prefix sequence: prefillWithLogits
// and the decodeStepLogits calls that follow it record into the same emitter, instead
// of each call opening (and prefill closing) its own. Every record carries the step's
// phase, absolute position (decode) and sequence step.

/** @type {typeof import('./diagnostic-sequence.js').beginDiagnosticSequence} */
export function beginDiagnosticSequence(state, options = {}) {
  if (state.isGenerating) throw new Error('A diagnostic sequence cannot begin during generation.');
  if (state.diagnosticSequence) throw new Error('A diagnostic sequence is already open.');
  const captureConfig = {
    ...createDefaultCaptureConfig(),
    enabled: true,
    defaultLevel: CAPTURE_LEVELS.SLICE,
    ...(options.captureConfig ?? {}),
  };
  validateCaptureConfig(captureConfig);
  state.operatorDiagnostics = {
    enabled: true,
    captureConfig,
    emitter: new OperatorEventEmitter({
      modelHash: state.manifest?.modelId ?? null,
      runtimeConfigHash: state.resolvedKernelPath?.id ?? null,
      executionPlanHash: null,
    }),
    tsirFixture: createTsirFixtureState(null),
  };
  state.diagnosticSequence = { steps: 0 };
}

/** @type {typeof import('./diagnostic-sequence.js').markDiagnosticSequenceStep} */
export function markDiagnosticSequenceStep(state, phase, options) {
  const sequence = state.diagnosticSequence;
  if (!sequence) return false;
  if (options?.diagnostics) {
    throw new Error('Per-call diagnostics cannot be supplied inside an open diagnostic sequence.');
  }
  const emitter = state.operatorDiagnostics?.emitter;
  if (!emitter) throw new Error('The open diagnostic sequence lost its operator diagnostics.');
  emitter.setContext({
    phase,
    tokenIndex: phase === 'decode' ? state.currentSeqLen : null,
    sequenceStep: sequence.steps,
  });
  sequence.steps += 1;
  return true;
}

/** @type {typeof import('./diagnostic-sequence.js').endDiagnosticSequence} */
export function endDiagnosticSequence(state) {
  if (!state.diagnosticSequence) throw new Error('No diagnostic sequence is open.');
  const emitter = state.operatorDiagnostics?.emitter ?? null;
  const result = {
    enabled: true,
    steps: state.diagnosticSequence.steps,
    timeline: emitter ? emitter.getTimeline() : [],
    recordCount: emitter ? emitter.length : 0,
  };
  state.operatorDiagnostics = null;
  state.diagnosticSequence = null;
  return result;
}

// One prefill-family call's own diagnostics session (no sequence open).
/** @type {typeof import('./diagnostic-sequence.js').beginPrefillDiagnostics} */
export function beginPrefillDiagnostics(state, options, executionPlan, tsirFixtureCfg = null) {
  state.operatorDiagnostics = null;
  state.stats.operatorDiagnostics = null;
  const enabled = options?.diagnostics?.enabled === true
    || state.runtimeConfig?.shared?.harness?.mode === 'diagnose';
  if (!enabled && !tsirFixtureCfg) return;
  const captureConfig = {
    ...createDefaultCaptureConfig(),
    enabled: true,
    defaultLevel: CAPTURE_LEVELS.SLICE,
    ...(options?.diagnostics?.captureConfig ?? {}),
  };
  validateCaptureConfig(captureConfig);
  state.operatorDiagnostics = {
    enabled,
    captureConfig,
    emitter: enabled ? new OperatorEventEmitter({
      modelHash: state.manifest?.modelId ?? null,
      runtimeConfigHash: state.resolvedKernelPath?.id ?? null,
      executionPlanHash: executionPlan?.planId ?? null,
    }) : null,
    tsirFixture: createTsirFixtureState(tsirFixtureCfg),
  };
}

/** @type {typeof import('./diagnostic-sequence.js').closePrefillDiagnostics} */
export function closePrefillDiagnostics(pipeline, options) {
  const state = pipeline._state;
  try {
    pipeline._closeFinitenessFallbackWindow(options);
  } finally {
    const emitter = state.operatorDiagnostics?.emitter;
    state.stats.operatorDiagnostics = emitter
      ? { enabled: true, timeline: emitter.getTimeline(), recordCount: emitter.length }
      : null;
    state.operatorDiagnostics = null;
  }
}
