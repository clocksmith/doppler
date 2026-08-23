export function resolveEmbeddingNormalization(value) {
  if (value == null) return null;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('output.embeddingNormalization must be null or an object');
  }
  if (value.type !== 'rmsnorm' || value.withScale !== false) {
    throw new Error('output.embeddingNormalization supports only weightless rmsnorm');
  }
  if (!Number.isFinite(value.eps) || value.eps <= 0) {
    throw new Error('output.embeddingNormalization.eps must be a positive finite number');
  }
  if (value.position !== 'after-scale') {
    throw new Error('output.embeddingNormalization.position must be "after-scale"');
  }
  return Object.freeze({
    type: 'rmsnorm',
    withScale: false,
    eps: value.eps,
    position: 'after-scale',
  });
}
