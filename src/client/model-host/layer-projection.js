import { runPipelineOperation } from '../../inference/pipelines/shader-scoped-pipeline.js';
import {
  LAYER_PROJECTION_MAX_ROWS,
  LAYER_PROJECTION_MAX_TOP_K,
  LAYER_PROJECTION_SCHEMA,
  topCandidates,
} from '../inspection.js';

// Projects recorded layer outputs through the loaded model's final
// normalization and output head. The pipeline owns the computation; this
// validates the request, snapshots it and labels the result. The candidates are
// a projection of each vector: they are not tokens the model produced there.
/** @type {typeof import('./layer-projection.js').projectLayerOutputs} */
export const projectLayerOutputs = async function projectLayerOutputs(pipeline, modelId, request) {
  const vectors = request?.vectors;
  const topK = request?.topK;
  if (!Array.isArray(vectors) || vectors.length === 0 || vectors.length > LAYER_PROJECTION_MAX_ROWS) {
    throw new Error(`Doppler model.inspect.projectLayerOutputs requires 1 to ${LAYER_PROJECTION_MAX_ROWS} vectors.`);
  }
  if (!Number.isInteger(topK) || topK < 1 || topK > LAYER_PROJECTION_MAX_TOP_K) {
    throw new Error(`Doppler model.inspect.projectLayerOutputs requires topK as a whole number from 1 to ${LAYER_PROJECTION_MAX_TOP_K}.`);
  }
  const width = vectors[0]?.length;
  if (!Number.isInteger(width) || width < 1 || vectors.some((vector) => !(vector instanceof Float32Array) || vector.length !== width)) {
    throw new Error('Doppler model.inspect.projectLayerOutputs requires Float32Array vectors of one length.');
  }
  // Snapshot caller-owned vectors before awaiting the pipeline's execution slot.
  const hiddenStates = new Float32Array(vectors.length * width);
  vectors.forEach((vector, row) => hiddenStates.set(vector, row * width));
  return runPipelineOperation(pipeline, async (scoped) => {
    if (typeof scoped.projectHiddenStates !== 'function') {
      throw new Error('The loaded Doppler pipeline cannot project hidden states.');
    }
    const { logits, rowCount, vocabSize } = await scoped.projectHiddenStates(hiddenStates, vectors.length);
    return {
      schema: LAYER_PROJECTION_SCHEMA,
      modelId,
      projection: { through: ['final-normalization', 'output-head'], kind: 'projected' },
      vocabSize,
      rows: Array.from({ length: rowCount }, (_, row) => ({
        candidates: topCandidates(logits.subarray(row * vocabSize, (row + 1) * vocabSize), scoped.tokenizer, topK),
      })),
    };
  });
};
