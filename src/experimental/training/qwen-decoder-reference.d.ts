import type { qwenFrozenLoraProjectionForward, qwenRmsNormOffsetForward } from './qwen-full-attention-reference.js';
type Projection = ReturnType<typeof qwenFrozenLoraProjectionForward>;
type Norm = ReturnType<typeof qwenRmsNormOffsetForward>;
export interface DecoderInputs {
  hidden: Float32Array;
  inputNormWeight: Float32Array;
  postAttentionNormWeight: Float32Array;
  attention: Record<string, unknown>;
  mlp: {
    gateWeight: Float32Array;
    upWeight: Float32Array;
    downWeight: Float32Array;
    lora?: { gate?: unknown; up?: unknown; down?: unknown };
  };
}
export interface DecoderOptions {
  numTokens: unknown;
  hiddenSize: unknown;
  intermediateSize: unknown;
  rmsEps: unknown;
}
export interface AttentionResult { output: Float32Array; cache: unknown }
export interface DecoderCache<A extends AttentionResult> {
  inputNorm: Norm;
  attentionInputs: Record<string, unknown>;
  attention: A;
  postAttention: Float32Array;
  normalizedPostAttention: Norm;
  gate: Projection;
  up: Projection;
  activated: Float32Array;
  down: Projection;
}
export declare function decoderLayerForward<A extends AttentionResult>(
  inputs: DecoderInputs, options: DecoderOptions,
  attentionForward: (inputs: Record<string, unknown>, options: DecoderOptions) => A
): { output: Float32Array; cache: DecoderCache<A> };
type LoraGradient = { A: Float32Array | null; B: Float32Array | null };
export declare function decoderLayerBackward<G extends { hidden: Float32Array }>(
  inputs: DecoderInputs, gradOutput: Float32Array, cache: DecoderCache<AttentionResult>, options: DecoderOptions,
  attentionBackward: (inputs: Record<string, unknown>, gradient: Float32Array, cache: unknown, options: DecoderOptions) => G
): { hidden: Float32Array; attentionGradients: G; lora: { gate: LoraGradient; up: LoraGradient; down: LoraGradient } };
