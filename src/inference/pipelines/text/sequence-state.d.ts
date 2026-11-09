import type { PipelineState } from './state.js';
type ContextOptions = import('./generator-runtime.js').GenerateOptionsResolved & import('./types.js').GenerateOptions;
export interface GenerationContext {
  getStats(): Readonly<{ allocatedBytes: number; cachedTokens: number; reusedTokens: number; status: string }>;
  close(): void;
}
export interface ContextEntry {
  owner: PipelineState;
  maxBytes: number;
  cache: PipelineState['kvCache'];
  linear: PipelineState['linearAttentionRuntime'];
  tokens: number[];
  bytes: number;
  signature: string | null;
  adapter?: PipelineState['lora'];
  busy: boolean;
  closed: boolean;
  reusedTokens: number;
  status: string;
}
export interface ContextLease {
  entry: ContextEntry;
  previous: { cache: PipelineState['kvCache']; linear: PipelineState['linearAttentionRuntime']; seqLen: number };
  signature: string;
  checkpoint?: { cache: PipelineState['kvCache']; linear: PipelineState['linearAttentionRuntime']; tokens: number[] };
}
export function supportsGenerationContext(state: PipelineState): boolean;
export function createGenerationContext(state: PipelineState, options: { maxBytes: number }): GenerationContext;
export function closeGenerationContexts(state: PipelineState): void;
export function beginGenerationContext(state: PipelineState, context: GenerationContext | undefined, options: ContextOptions): void;
export function generationContextInput(state: PipelineState, inputIds: number[]): number[];
export function generationContextCheckpointLength(state: PipelineState, prompt: import('./generator.js').PromptInput,
  inputIds: number[], options: ContextOptions): number | null;
export function checkpointGenerationContext(state: PipelineState, tokens: number[]): Promise<void>;
export function prepareGenerationContextPrefill(generator: {
  _state: PipelineState;
  _resolvePrefillTokenChunkSize(ids: number[]): number | null;
  _prefillToHidden(ids: number[], options: ContextOptions): Promise<unknown>;
  _commitPrefillHiddenChunk(value: unknown): Promise<void>;
}, prompt: import('./generator.js').PromptInput, inputIds: number[], options: ContextOptions): Promise<number | null>;
export function finishGenerationContext(state: PipelineState, generatedIds: number[], completed: boolean): Promise<void>;

export interface SequenceState {
  isGenerating: boolean;
  currentSeqLen: number;
  kvCache?: { truncate?(seqLen: number): void } | null;
}

/** Rejects non-number, fractional, negative, unsafe, and increasing lengths before mutation. */
export function resetSequenceState(state: SequenceState, seqLen: number): void;
