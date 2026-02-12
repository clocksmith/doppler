export function biasAddRef(data, bias, numTokens, dim) {
  const output = new Float32Array(data.length);
  for (let t = 0; t < numTokens; t++) {
    const base = t * dim;
    for (let d = 0; d < dim; d++) {
      output[base + d] = data[base + d] + bias[d];
    }
  }
  return output;
}

export function biasAddBackwardRef(gradOutput, numTokens, dim) {
  const gradBias = new Float32Array(dim);
  for (let t = 0; t < numTokens; t++) {
    for (let d = 0; d < dim; d++) {
      gradBias[d] += gradOutput[t * dim + d];
    }
  }
  return gradBias;
}
