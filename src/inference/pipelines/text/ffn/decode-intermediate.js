import { isTraceEnabled } from '../../../../debug/index.js';

/** @type {import('./decode-intermediate.js').borrowDecodeFfnIntermediate} */
export function borrowDecodeFfnIntermediate(context, numTokens, intermediateSize, dtype) {
  if (context.phase !== 'decode' || numTokens !== 1 || dtype !== 'f32'
      || context.activationPrecision === 'two-f32' || context.trainingCapture || context.operatorDiagnostics
      || (context.debugProbes?.length ?? 0) > 0 || isTraceEnabled('logits')) return null;
  const buffer = context.decodeBuffers?.getFFNIntermediateBuffer();
  return buffer && buffer.size >= intermediateSize * Float32Array.BYTES_PER_ELEMENT ? buffer : null;
}
