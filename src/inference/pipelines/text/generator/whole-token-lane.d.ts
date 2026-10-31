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
