export function modulateRef(input, mod, numTokens, hiddenSize, options) {
  const {
    scaleOffset = 0,
    shiftOffset = hiddenSize,
    gateOffset = hiddenSize * 2,
    hasGate = false,
    addOne = true,
  } = options;
  const output = new Float32Array(numTokens * hiddenSize);
  for (let t = 0; t < numTokens; t++) {
    const base = t * hiddenSize;
    for (let d = 0; d < hiddenSize; d++) {
      const rawScale = mod[scaleOffset + d];
      const shift = mod[shiftOffset + d];
      const scale = addOne ? 1 + rawScale : rawScale;
      let value = input[base + d] * scale + shift;
      if (hasGate) {
        value *= mod[gateOffset + d];
      }
      output[base + d] = value;
    }
  }
  return output;
}
