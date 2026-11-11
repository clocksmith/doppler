import type { DopplerLayerProjection } from '../inspection.js';

/** Validates a projection request, runs it in the pipeline's execution slot and labels the result. */
export declare function projectLayerOutputs(
  pipeline: {
    projectHiddenStates?: (hiddenStates: Float32Array, rowCount: number) => Promise<{ logits: Float32Array; rowCount: number; vocabSize: number }>;
    tokenizer: { decode(ids: number[], skipSpecialTokens?: boolean, trim?: boolean): string };
  },
  modelId: string,
  request: { vectors: Float32Array[]; topK: number }
): Promise<DopplerLayerProjection>;
