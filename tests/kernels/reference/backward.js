export function softmaxBackwardRef(softmax, gradOutput, rows, cols) {
  const output = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    const base = r * cols;
    let sum = 0;
    for (let c = 0; c < cols; c++) {
      sum += softmax[base + c] * gradOutput[base + c];
    }
    for (let c = 0; c < cols; c++) {
      const idx = base + c;
      output[idx] = softmax[idx] * (gradOutput[idx] - sum);
    }
  }
  return output;
}

export function siluBackwardRef(input, gradOutput) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const sigmoid = 1 / (1 + Math.exp(-x));
    const deriv = sigmoid * (1 + x * (1 - sigmoid));
    output[i] = gradOutput[i] * deriv;
  }
  return output;
}

export function geluBackwardRef(input, gradOutput) {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const x = input[i];
    const sqrt2pi = 0.7978845608;
    const c = 0.044715;
    const x3 = x * x * x;
    const inner = sqrt2pi * (x + c * x3);
    const innerClamped = Math.max(-15, Math.min(15, inner));
    const tanhInner = Math.tanh(innerClamped);
    const sech2 = 1 - tanhInner * tanhInner;
    const innerDeriv = sqrt2pi * (1 + 3 * c * x * x);
    const deriv = 0.5 * (1 + tanhInner) + 0.5 * x * sech2 * innerDeriv;
    output[i] = gradOutput[i] * deriv;
  }
  return output;
}

export function scaleBackwardRef(gradOutput, scale) {
  const output = new Float32Array(gradOutput.length);
  for (let i = 0; i < gradOutput.length; i++) {
    output[i] = gradOutput[i] * scale;
  }
  return output;
}

export function ropeBackwardRef(gradOutput, cos, sin, seqLen, numHeads, headDim, startPos) {
  const output = new Float32Array(gradOutput.length);
  const halfDim = headDim / 2;
  for (let pos = 0; pos < seqLen; pos++) {
    for (let h = 0; h < numHeads; h++) {
      const base = (pos * numHeads + h) * headDim;
      const freqBase = (startPos + pos) * halfDim;
      for (let i = 0; i < halfDim; i++) {
        const dy0 = gradOutput[base + i];
        const dy1 = gradOutput[base + i + halfDim];
        const co = cos[freqBase + i];
        const s = sin[freqBase + i];
        output[base + i] = dy0 * co + dy1 * s;
        output[base + i + halfDim] = -dy0 * s + dy1 * co;
      }
    }
  }
  return output;
}

export function rmsNormBackwardRef(input, weight, gradOutput, numTokens, hiddenSize, eps) {
  const output = new Float32Array(numTokens * hiddenSize);
  for (let t = 0; t < numTokens; t++) {
    const base = t * hiddenSize;
    let sumSq = 0;
    let sumGX = 0;
    for (let i = 0; i < hiddenSize; i++) {
      const x = input[base + i];
      const g = gradOutput[base + i] * weight[i];
      sumSq += x * x;
      sumGX += g * x;
    }
    const invRms = 1 / Math.sqrt(sumSq / hiddenSize + eps);
    const invRms3 = invRms * invRms * invRms;
    const coeff = (sumGX / hiddenSize) * invRms3;
    for (let i = 0; i < hiddenSize; i++) {
      const x = input[base + i];
      const g = gradOutput[base + i] * weight[i];
      output[base + i] = g * invRms - x * coeff;
    }
  }
  return output;
}
