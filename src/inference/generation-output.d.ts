import type { ReasoningOutputContract } from '../config/reasoning-output.js';
import type { IncrementalTokenDecoder } from './tokenizers/bundled/incremental-decoder.js';
export interface OutputTokenizer {
  decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string;
  createIncrementalDecoder(): IncrementalTokenDecoder;
  createRawIncrementalDecoder?(): IncrementalTokenDecoder;
}
export function createGenerationOutputDecoder(tokenizer: OutputTokenizer,
  contract: ReasoningOutputContract | null | undefined, stopTokenIds?: readonly number[]): IncrementalTokenDecoder;
export function validateGenerationOutputTokenizer(tokenizer: OutputTokenizer,
  contract: ReasoningOutputContract | null | undefined, stopTokenIds?: readonly number[]): ReasoningOutputContract | null | undefined;
export function decodeGenerationOutput(tokenizer: OutputTokenizer,
  contract: ReasoningOutputContract | null | undefined, stopTokenIds: readonly number[],
  tokenIds: number[], complete?: boolean, stopReason?: string | null): string;
export function decodeGenerationStream(tokens: AsyncIterable<number>, decoder: IncrementalTokenDecoder,
  options?: { signal?: AbortSignal; onToken?: (id: number, text: string) => void;
    getStopReason?: () => string | null | undefined }): AsyncGenerator<string>;
