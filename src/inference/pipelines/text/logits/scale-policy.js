export function resolveLogitInputScale(config) {
  const value = config?.logitInputScale;
  if (value == null) {
    throw new Error('[Logits] logitInputScale is required.');
  }
  const scale = Number(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`[Logits] logitInputScale must be a positive finite number; got "${String(value)}".`);
  }
  return scale;
}

export function resolveLogitOutputScale(config) {
  const value = config?.logitOutputScale ?? 1;
  const scale = Number(value);
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error(`[Logits] logitOutputScale must be a positive finite number; got "${String(value)}".`);
  }
  return scale;
}
