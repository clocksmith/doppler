export function embedBackwardRef(indicesOrGradOutput, gradOutput, numTokens, hiddenSize, vocabSize) {
  if (arguments.length === 1 || gradOutput === undefined) {
    return new Float32Array(indicesOrGradOutput);
  }

  const indices = indicesOrGradOutput;
  const out = new Float32Array(vocabSize * hiddenSize);
  for (let i = 0; i < numTokens; i++) {
    const token = indices[i];
    for (let d = 0; d < hiddenSize; d++) {
      out[token * hiddenSize + d] += gradOutput[i * hiddenSize + d];
    }
  }
  return out;
}
