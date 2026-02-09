export function crossEntropyLossRef(softmax, targets, numTokens, vocabSize) {
  const output = new Float32Array(numTokens);
  for (let t = 0; t < numTokens; t++) {
    const target = targets[t];
    if (target >= vocabSize) {
      output[t] = 0;
      continue;
    }
    const p = Math.max(softmax[t * vocabSize + target], 1e-9);
    output[t] = -Math.log(p);
  }
  return output;
}

export function crossEntropyBackwardRef(softmax, targets, gradOutput, numTokens, vocabSize) {
  const output = new Float32Array(numTokens * vocabSize);
  for (let t = 0; t < numTokens; t++) {
    const target = targets[t];
    for (let c = 0; c < vocabSize; c++) {
      let grad = softmax[t * vocabSize + c];
      if (c === target) grad -= 1;
      output[t * vocabSize + c] = grad * gradOutput[t];
    }
  }
  return output;
}
