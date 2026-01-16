


export function residualAddRef(x, residual) {
  const output = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i];
  }

  return output;
}


export function residualAddInplaceRef(x, residual) {
  for (let i = 0; i < x.length; i++) {
    x[i] += residual[i];
  }
  return x;
}


export function scaledResidualAddRef(x, residual, scale) {
  const output = new Float32Array(x.length);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + scale * residual[i];
  }

  return output;
}


export function residualAddDropoutRef(x, residual, mask, dropProb) {
  const output = new Float32Array(x.length);
  const scale = 1.0 / (1.0 - dropProb);

  for (let i = 0; i < x.length; i++) {
    output[i] = x[i] + residual[i] * mask[i] * scale;
  }

  return output;
}

export default residualAddRef;
