import type { GreedySpeculativeSession, GreedyFrameProgram, GreedyFrameTransaction } from '../../../../gpu/greedy-speculative-frame.js';
import type { LinearAttentionRuntime } from '../linear-attention.js';
import type { PositionLimit } from './whole-token-observation.js';

export interface CompiledGreedyProgram extends GreedyFrameProgram {
  /** Set tentative KV extents and all host cursors for a replay that executed maxTokens rows. */
  setReplayedPrefix(count: number): void | Promise<void>;
}
export interface CompiledGreedyWindow {
  readonly ready: boolean;
  readonly stats: { observedSteps: number; replayedTokens: number; replaySubmissions: number; privateBytes: number };
  begin(program: CompiledGreedyProgram, position: number, outputLimit: number, signal?: AbortSignal): Promise<GreedyFrameTransaction>;
  /** Session is borrowed. Dispose this controller before disposing that session. */
  dispose(): Promise<void>;
}
/** Internal greedy controller; caller retains exclusive session/model ownership until dispose. */
export function createCompiledGreedyWindow(options: {
  device: GPUDevice;
  session: GreedySpeculativeSession;
  runtime: LinearAttentionRuntime;
  draftCount: number;
  vocabSize: number;
  hiddenSize: number;
  identityOf(): readonly unknown[];
  positionLimits: readonly PositionLimit[];
}): CompiledGreedyWindow;
