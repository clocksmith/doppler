import type { PipelineState } from './state.js';
import type { AnyLayerPartitionPlan } from './layer-partition-contract.js';
export type ContextOptions = import('./generator-runtime.js').GenerateOptionsResolved & import('./types.js').GenerateOptions;
export type ContextPipelineState = PipelineState & {
  modelPartition?: { plan: AnyLayerPartitionPlan; index: number } | null;
};
export interface HyperConnectionSnapshot {
  ple: Map<number, { conv: GPUBuffer }>;
  qsa: Map<number, { length: number; rawKeys: GPUBuffer | null; blockKeys: GPUBuffer | null }>;
}
export interface GenerationContext {
  getStats(): Readonly<{ allocatedBytes: number; cachedTokens: number; reusedTokens: number; status: string }>;
  close(): void;
}
export interface ContextEntry {
  owner: ContextPipelineState;
  maxBytes: number;
  cache: PipelineState['kvCache'];
  linear: PipelineState['linearAttentionRuntime'];
  conv: Map<number, GPUBuffer> | null;
  hyper: HyperConnectionSnapshot | null;
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
  conv: Map<number, GPUBuffer> | null;
  hyper: HyperConnectionSnapshot | null;
  checkpoint?: {
    cache: PipelineState['kvCache'];
    linear: PipelineState['linearAttentionRuntime'];
    conv: Map<number, GPUBuffer> | null;
    hyper: HyperConnectionSnapshot | null;
    tokens: number[];
  };
}
export function supportsGenerationContext(state: ContextPipelineState): boolean;
export function createGenerationContext(state: ContextPipelineState, options: { maxBytes: number }): GenerationContext;
export function closeGenerationContexts(state: ContextPipelineState): void;
/** Retire idle pooled memory before a new generation's first prefill when a device budget is set. */
export function beginSequenceMemoryBoundary(): Promise<boolean>;
export function beginGenerationContext(state: ContextPipelineState, context: GenerationContext | undefined, options: ContextOptions): void;
export function generationContextInput(state: ContextPipelineState, inputIds: number[]): number[];
export function generationContextCheckpointLength(state: ContextPipelineState, prompt: import('./generator.js').PromptInput,
  inputIds: number[], options: ContextOptions): number | null;
export function checkpointGenerationContext(state: ContextPipelineState, tokens: number[]): Promise<void>;
export function prepareGenerationContextPrefill(generator: {
  _state: ContextPipelineState;
  _resolvePrefillTokenChunkSize(ids: number[]): number | null;
  _prefillToHidden(ids: number[], options: ContextOptions): Promise<unknown>;
  _commitPrefillHiddenChunk(value: unknown): Promise<void>;
}, prompt: import('./generator.js').PromptInput, inputIds: number[], options: ContextOptions): Promise<number | null>;
export function finishGenerationContext(state: ContextPipelineState, generatedIds: number[], completed: boolean): Promise<void>;

export interface SequenceState {
  isGenerating: boolean;
  currentSeqLen: number;
  kvCache?: { truncate?(seqLen: number): void } | null;
}

/** Rejects non-number, fractional, negative, unsafe, and increasing lengths before mutation. */
export function resetSequenceState(state: SequenceState, seqLen: number): void;
