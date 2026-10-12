import type { IncrementalTokenDecoder } from './tokenizers/bundled/incremental-decoder.js';
import type { CanonicalGenerationPolicy, CanonicalStopReason } from '../config/generation-contract.js';
export interface OutputControl {
  withheld: string; rawUtf16: number; visibleUtf16: number; tokens: number; stopped: boolean; closed: boolean;
}
export interface ResidentOutputState {
  accept(tokenId: number): { delta: string; textStop: boolean };
  finalize(reason: CanonicalStopReason): string;
  getUsage(): OutputControl;
  digest(): string;
  clone(): ResidentOutputState;
}
export function createResidentOutputState(raw: IncrementalTokenDecoder, visible: IncrementalTokenDecoder,
  policy: CanonicalGenerationPolicy, vocabSize: number): ResidentOutputState;
