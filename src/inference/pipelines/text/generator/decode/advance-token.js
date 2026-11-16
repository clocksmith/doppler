import { getActivationRepresentation } from "../session/activation-storage.js";
import { releaseBuffer } from "../../../../../memory/buffer-pool.js";
import { allowReadback } from "../../../../../debug/perf-guards.js";
import { extractEmbeddingFromHiddenGPU } from "../../embedding/embedding-extraction.js";
import { advanceDecodeStepCount } from "../../observation/tsir-fixture-writer.js";
import { getEffectiveActivationDtype, isOwnedDecodeBuffer, runDecodeLayers } from "./decode.js";

export async function advanceWithToken(state, tokenId, opts, helpers) {
  advanceDecodeStepCount(state);

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer } = await runDecodeLayers(
    state,
    tokenId,
    opts,
    helpers
  );

  const isPreAllocated = isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer);
  if (!isPreAllocated) {
    releaseBuffer(hiddenStates);
  }

  state.currentSeqLen++;
}

export async function advanceWithTokenAndEmbedding(state, tokenId, opts, helpers, embeddingMode) {

  advanceDecodeStepCount(state);

  const { hiddenStates, decodeHiddenBuffer, decodeAltBuffer } = await runDecodeLayers(
    state,
    tokenId,
    opts,
    helpers
  );

  if (!allowReadback('pipeline.advance.embedding')) {
    throw new Error('GPU readback disabled; cannot return embedding');
  }

  const config = state.modelConfig;
  const activationDtype = getEffectiveActivationDtype(state, opts);

  let embedding;
  try {
    const extracted = await extractEmbeddingFromHiddenGPU({
      hiddenBuffer: hiddenStates,
      activationDtype,
      representation: getActivationRepresentation(state, opts.executionPlan),
      numTokens: 1,
      hiddenSize: config.hiddenSize,
      embeddingMode,
      finalNorm: state.weights.get('final_norm'),
      finalNormBias: state.weights.get('final_norm_bias') ?? null,
      config,
      embeddingPostprocessor: state.embeddingPostprocessor,
    });
    embedding = extracted.embedding;
  } finally {
    const isPreAllocated = isOwnedDecodeBuffer(hiddenStates, decodeHiddenBuffer, decodeAltBuffer);
    if (!isPreAllocated) {
      releaseBuffer(hiddenStates);
    }
  }

  state.currentSeqLen++;

  return {
    embedding,
    embeddingMode,
    seqLen: state.currentSeqLen,
  };
}
