export function validateRoPEInverseFrequencies(value, rotaryDim, label) {
  if (value === null) return null;
  if (!Array.isArray(value) || value.length !== rotaryDim / 2
    || value.some((entry) => typeof entry !== 'number' || !Number.isFinite(entry)
      || entry <= 0 || Math.fround(entry) !== entry)) {
    throw new Error(`${label} must be null or ${rotaryDim / 2} positive, finite f32 inverse frequencies.`);
  }
  return [...value];
}
