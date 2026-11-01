import type { CommandRecorder } from '../../../../gpu/command-recorder.js';
import type { PipelineState } from '../state.js';

/**
 * Whether decodeLoop.wholeTokenProgram is requested for this step; throws when it is
 * requested together with an input the program cannot replay (debug, profiling,
 * observers, logit masking, penalties, stochastic sampling, per-layer inputs).
 */
export interface WholeTokenStepOptions {
  debug?: boolean;
  profile?: boolean;
  executionObserver?: unknown;
  onLogits?: unknown;
  logitMaskFn?: unknown;
  suppressTokenIds: number[];
  repetitionPenalty?: number;
  presencePenalty?: number;
  temperature: number;
  topK: number;
}
export declare function wholeTokenProgramRequested(state: PipelineState, opts: WholeTokenStepOptions): boolean;

/**
 * One decode step of the whole-token lane. `runRecorded` runs the ordinary recorded
 * step; given an `attach` function, it hands that the step's recorder first.
 */
export declare function decodeWholeTokenStep(
  state: PipelineState,
  currentIds: number[],
  runRecorded: (attach: ((recorder: CommandRecorder | null) => void) | null) => Promise<number>
): Promise<number>;

/** Thrown by a decode step recorded for verification ahead, in place of its submit. */
export declare const WHOLE_TOKEN_DRY_STEP_STOP: Readonly<{ reason: string }>;

/**
 * Verify the kept program on decode step 1 before it runs (WholeTokenProgram.verifyAhead):
 * `runRecordedDry` records the ordinary step at `position` and stops before its submit.
 * CPU sequence bookkeeping the step advanced is restored. True when step 1 at `position`
 * replays the program; false without a kept program or when it was invalidated.
 */
export declare function verifyWholeTokenProgramAhead(
  state: PipelineState,
  position: number,
  token: number,
  runRecordedDry: (attach: ((recorder: CommandRecorder | null) => void) | null) => Promise<unknown>
): Promise<boolean>;
