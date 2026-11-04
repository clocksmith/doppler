import type { CommandRecorder } from './command-recorder.js';
import type { Tensor } from './tensor.js';

export interface GreedyFrameRows { hidden: Tensor; logits: Tensor }
export interface GreedyFrameRefillShape { readonly rows: number; readonly width: number }
export interface GreedyFrameProgram {
  /** Record only. Returned outputs transfer to the frame recorder. */
  recordDraft(recorder: CommandRecorder, tokenIds: GPUBuffer, tokenIndex: number,
    hidden: Tensor, index: number): Promise<GreedyFrameRows>;
  /** Input IDs are GPU argmax results. All target rows must execute in this recorder. */
  recordTarget(recorder: CommandRecorder, tokenIds: GPUBuffer, count: number): Promise<GreedyFrameRows>;
  /** Refill the private auxiliary cache; transfer all outputs matching the session's declared shapes. */
  recordRefill(recorder: CommandRecorder, hidden: Tensor, targetIds: GPUBuffer): Promise<Tensor[]>;
  /** Restore/truncate CPU cursors to initial position + count, never increment them. */
  setPrefix(count: number): void | Promise<void>;
}
export interface GreedyFrameTransaction {
  readonly tokens: readonly number[];
  readonly accepted: number;
  /** Mark exactly one token as delivered, immediately before application delivery. */
  publish(): void;
  /** Idempotent; recover precisely the published prefix, retaining all GPU leases until done. */
  settle(): Promise<void>;
}
export interface GreedySpeculativeSession {
  begin(program: GreedyFrameProgram, outputLimit: number, signal?: AbortSignal): Promise<GreedyFrameTransaction>;
  dispose(): Promise<void>;
}
/** Internal F32, greedy, recurrent-model execution mechanism. Not a public generation API.
 * Caller holds exclusive model/cache ownership for the entire session, including initialization.
 */
export function createGreedySpeculativeSession(options: {
  owner: object; hiddenSize: number; vocabSize: number; draftCount: number;
  /** Draft scores cover global IDs [0, draftVocabSize); target verification remains full vocabulary. */
  draftVocabSize: number;
  /** Every computed refill output is finite-checked, including KV-only programs that compute no hidden rows. */
  refillShapes: readonly GreedyFrameRefillShape[];
  initialHidden: Tensor; initialToken: number; padTokenId: number | null;
  finitenessAbsThreshold: number; usePreparedDecodeWorkspace: boolean;
}): Promise<GreedySpeculativeSession>;
