import type { GreedySpeculativeSession, GreedyFrameProgram, GreedyFrameTransaction } from '../../../../gpu/greedy-speculative-frame.js';
import type { LinearAttentionRuntime } from '../linear-attention.js';
import type { PositionLimit } from './whole-token-observation.js';
import type { Tensor } from '../../../../gpu/tensor.js';

export interface CompiledGreedyProgram extends GreedyFrameProgram {
  /** Set tentative KV extents and all host cursors for a replay that executed maxTokens rows. */
  setReplayedPrefix(count: number): void | Promise<void>;
}
export interface CompiledGreedyWindow {
  readonly ready: boolean;
  /** privateBytes covers compiled resources, excluding model weights, caches and the rollback checkpoint. */
  readonly stats: { observedSteps: number; replayedTokens: number; replaySubmissions: number; privateBytes: number; reused: boolean; verifiedAhead: boolean };
  /** Keep bindings, reset carry and require dry-recorded plan/resource verification before the next replay. */
  beginGeneration(initialHidden: Tensor, initialToken: number): Promise<void>;
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
