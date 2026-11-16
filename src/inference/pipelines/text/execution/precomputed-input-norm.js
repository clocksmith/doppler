import { releaseOrTrack } from './ops.js';

/** @type {typeof import('./precomputed-input-norm.js').releasePrecomputedInputNorm} */
export function releasePrecomputedInputNorm(context, recorder) {
  const precomputed = context.__precomputedInputNorm ?? null;
  context.__precomputedInputNorm = null;
  const buffer = precomputed?.tensor?.buffer ?? null;
  if (buffer) {
    releaseOrTrack(recorder, buffer, context.decodeBuffers);
  }
}
/** @type {typeof import('./precomputed-input-norm.js').takePrecomputedInputNorm} */
export function takePrecomputedInputNorm(context, layerIdx, recorder) {
  const precomputed = context.__precomputedInputNorm ?? null;
  if (!precomputed) {
    return null;
  }
  context.__precomputedInputNorm = null;
  if (precomputed.layerIdx !== layerIdx) {
    const buffer = precomputed?.tensor?.buffer ?? null;
    if (buffer) {
      releaseOrTrack(recorder, buffer, context.decodeBuffers);
    }
    throw new Error(
      `Layer ${layerIdx} received stale precomputed input norm for layer ${String(precomputed.layerIdx)}.`
    );
  }
  return precomputed.tensor;
}
