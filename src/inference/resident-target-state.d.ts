import type { CanonicalGenerationPolicy } from '../config/generation-contract.js';
import type { ResidentOutputState } from './resident-output-state.js';
import type { ResidentCursor, ResidentBlockDecision } from './resident-state-contract.js';

export interface ResidentTargetTransition {
  acceptedTokenIds: readonly number[];
  tokenRange: Readonly<{ start: number; end: number }>;
  successor: Readonly<ResidentCursor>;
  delta: string;
  outputDigest: string;
}
export interface ResidentTargetState {
  getCursor(): Readonly<ResidentCursor> | null;
  getHistory(): readonly number[];
  bootstrap(logits: Float32Array): Readonly<ResidentTargetTransition>;
  advance(expectedCursorId: string, logits: Float32Array): Readonly<ResidentTargetTransition>;
  /** Local control-state clone. The resident must separately clone numerical state. */
  clone(): ResidentTargetState;
  /** Mutates this scratch control state only; never call on a published parent. */
  verifyBlock(expectedCursorId: string, proposal: readonly number[], rows: readonly Float32Array[]):
    Readonly<ResidentTargetTransition & { decision: Readonly<ResidentBlockDecision> }>;
}
export function createResidentTargetState(prompt: readonly number[], policy: CanonicalGenerationPolicy,
  vocabSize: number, generationId: string, output: ResidentOutputState): ResidentTargetState;
