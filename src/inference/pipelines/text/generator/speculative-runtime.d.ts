import type { PipelineState } from '../state.js';
import type { GenerateOptionsResolved } from '../generator-runtime.js';
export interface SpeculativeGenerator {
  _state: PipelineState & { modelPartition?: object | null };
  _prefill(tokens: number[], options: GenerateOptionsResolved & { _returnAllLogits?: boolean }): Promise<Float32Array>;
  _sampleNextTokenFromLogits(logits: Float32Array, context: number[], options: GenerateOptionsResolved): number;
}
/** Token lookup is a proposal only; every returned token is selected by the target. */
export function lookupDraft(context: readonly number[], maxTokens: number, maxMatchTokens: number, minMatchTokens: number): number[];
export function verifySpeculativeTokens(generator: SpeculativeGenerator, context: number[], proposal: number[],
  options: GenerateOptionsResolved, signal: AbortSignal | undefined,
  shouldStop: (context: number[], token: number) => boolean): Promise<{ tokens: number[]; accepted: number }>;

export function assertSpeculativeCapability(state: PipelineState & { modelPartition?: object | null }, options: { temperature: number; logitMaskFn?: unknown }): void;
