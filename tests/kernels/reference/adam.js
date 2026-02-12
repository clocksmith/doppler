export function adamRef(params, grads, moment1, moment2, options) {
  const { count, step, lr, beta1, beta2, eps } = options;
  const outParams = new Float32Array(params);
  const outM1 = new Float32Array(moment1);
  const outM2 = new Float32Array(moment2);
  for (let i = 0; i < count; i++) {
    const g = grads[i];
    outM1[i] = beta1 * outM1[i] + (1 - beta1) * g;
    outM2[i] = beta2 * outM2[i] + (1 - beta2) * g * g;
    const mHat = outM1[i] / (1 - Math.pow(beta1, step));
    const vHat = outM2[i] / (1 - Math.pow(beta2, step));
    outParams[i] = outParams[i] - lr * mHat / (Math.sqrt(vHat) + eps);
  }
  return { params: outParams, moment1: outM1, moment2: outM2 };
}
