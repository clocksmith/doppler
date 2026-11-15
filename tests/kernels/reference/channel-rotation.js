export function rotationReference(input, { channels, start, stop, contentStart, contentStop, epsilon }) {
  const first = new Float64Array(channels);
  const second = new Float64Array(channels);
  for (let token = contentStart; token < contentStop; token++) {
    for (let channel = 0; channel < channels; channel++) {
      const value = input[token * channels + channel];
      if (!Number.isFinite(value)) return { basis: new Float64Array(channels * 2), status: -1 };
      if (token >= start && token < stop) first[channel] += value / (stop - start);
      else second[channel] += value / (contentStop - contentStart - stop + start);
    }
  }
  const norm = values => Math.hypot(...values);
  const normA = norm(first);
  if (normA <= epsilon) return { basis: new Float64Array(channels * 2), status: 0 };
  for (let i = 0; i < channels; i++) first[i] /= normA;
  const projection = first.reduce((sum, x, i) => sum + x * second[i], 0);
  for (let i = 0; i < channels; i++) second[i] -= first[i] * projection;
  const normB = norm(second);
  if (normB <= epsilon) return { basis: new Float64Array(channels * 2), status: 0 };
  const basis = new Float64Array(channels * 2);
  basis.set(first);
  basis.set(second.map(x => x / normB), channels);
  return { basis, status: 1 };
}

export function applyRotationReference(input, { basis, status }, { channels, theta }) {
  if (status < 0) return new Float64Array(input.length).fill(NaN);
  const output = new Float64Array(input);
  if (status === 0 || theta === 0) return output;
  for (let offset = 0; offset < input.length; offset += channels) {
    let p = 0, q = 0;
    for (let i = 0; i < channels; i++) {
      p += basis[i] * input[offset + i];
      q += basis[channels + i] * input[offset + i];
    }
    const first = (Math.cos(theta) - 1) * p - Math.sin(theta) * q;
    const second = (Math.cos(theta) - 1) * q + Math.sin(theta) * p;
    for (let i = 0; i < channels; i++) output[offset + i] += basis[i] * first + basis[channels + i] * second;
  }
  return output;
}
