import type { ProbeConfigSchema } from '../../config/schema/index.js';
import type { LogitsConfig } from '../../inference/pipelines/text/logits/types.js';

export declare function extractLastPositionLogits(
  logits: Float32Array,
  numTokens: number,
  vocabSize: number
): Float32Array;

export declare function finalizeLogits(
  rawLogits: Float32Array,
  numTokens: number,
  matmulVocabSize: number,
  vocabSize: number,
  config: LogitsConfig,
  debugProbes?: ProbeConfigSchema[] | null,
  operatorDiagnostics?: unknown
): Promise<Float32Array>;
