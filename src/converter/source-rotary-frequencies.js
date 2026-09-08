export async function readSourceRotaryFrequencies(tensors, policy, readTensor) {
  if (policy == null) return null;
  if (typeof policy.match !== 'string' || !policy.match
    || !Number.isSafeInteger(policy.expectedMatches) || policy.expectedMatches <= 0) {
    throw new Error('sourceRotaryFrequencies requires match and positive expectedMatches.');
  }
  const pattern = new RegExp(policy.match);
  const matched = tensors.filter((tensor) => pattern.test(tensor.name));
  if (matched.length !== policy.expectedMatches) {
    throw new Error(`sourceRotaryFrequencies matched ${matched.length} tensors; expected ${policy.expectedMatches}.`);
  }
  let frequencies = null;
  for (const tensor of matched) {
    if (!['F32', 'f32'].includes(tensor.dtype) || tensor.shape?.length !== 1
      || !Number.isSafeInteger(tensor.shape[0]) || tensor.shape[0] <= 0
      || tensor.size !== tensor.shape[0] * Float32Array.BYTES_PER_ELEMENT) {
      throw new Error(`sourceRotaryFrequencies requires a contiguous f32 vector: ${tensor.name}.`);
    }
    const bytes = await readTensor(tensor);
    if (!(bytes instanceof ArrayBuffer) || bytes.byteLength !== tensor.size) {
      throw new Error(`sourceRotaryFrequencies byte length mismatch: ${tensor.name}.`);
    }
    const values = Array.from(new Float32Array(bytes));
    if (values.some((value) => !Number.isFinite(value) || value <= 0)) {
      throw new Error(`sourceRotaryFrequencies contains invalid values: ${tensor.name}.`);
    }
    if (frequencies && (values.length !== frequencies.length
      || values.some((value, index) => value !== frequencies[index]))) {
      throw new Error(`sourceRotaryFrequencies differ across layers: ${tensor.name}. Per-layer frequencies are unsupported.`);
    }
    frequencies = values;
  }
  return frequencies;
}
