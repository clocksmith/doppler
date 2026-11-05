import type { CommandRecorder } from '../command-recorder.js';
import type { Tensor } from '../tensor.js';

export interface PleRowDecodeOptions {
  numTokens: number;
  /** headCount * rowWidth BF16 values per token. */
  embedDim: number;
}

export interface PleGateOptions {
  numTokens: number;
  streamCount: number;
  streamWidth: number;
}

export interface PleConvOptions {
  numTokens: number;
  /** streamCount * streamWidth. */
  channels: number;
  kernel: number;
  dilation: number;
}

export interface PleGateResult {
  /** sigmoid(g') * value, [numTokens, streamCount * streamWidth]. */
  gated: Tensor;
  /** Signed square-root gate g' before the sigmoid, [numTokens, streamCount]. */
  gateInput: Tensor;
}

/** Decode gathered BF16 table rows ([numTokens, embedDim]) to f32. */
export declare function runPleRowDecode(rows: GPUBuffer, options: PleRowDecodeOptions): Promise<Tensor>;
export declare function recordPleRowDecode(recorder: CommandRecorder, rows: GPUBuffer, options: PleRowDecodeOptions): Promise<Tensor>;

export declare function runPleGate(keyNormed: Tensor, queryNormed: Tensor, value: Tensor, options: PleGateOptions): Promise<PleGateResult>;
export declare function recordPleGate(
  recorder: CommandRecorder, keyNormed: Tensor, queryNormed: Tensor, value: Tensor, options: PleGateOptions
): Promise<PleGateResult>;

/**
 * streams + (gated + silu(dilated depthwise conv of normed with the owned state));
 * the state ([(kernel - 1) * dilation, channels], oldest first) is updated in place.
 */
export declare function runPleConv(
  normed: Tensor, gated: Tensor, streams: Tensor, convWeight: Tensor, state: GPUBuffer, options: PleConvOptions
): Promise<Tensor>;
export declare function recordPleConv(
  recorder: CommandRecorder, normed: Tensor, gated: Tensor, streams: Tensor, convWeight: Tensor, state: GPUBuffer, options: PleConvOptions
): Promise<Tensor>;
