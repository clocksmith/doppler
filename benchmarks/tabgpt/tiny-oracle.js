// Independent CPU autograd expectations on a two-block model. The reference
// used PyTorch float64 math with FP32 master weights and Adam moment stores.
export async function checkTinyAutograd(createModel, fixtureUrl = new URL('./tiny-pytorch-oracle.json', import.meta.url)) {
  const response = await fetch(fixtureUrl);
  if (!response.ok) throw new Error('Tiny TabGPT CPU oracle is unavailable');
  const fixture = await response.json();
  const model = await createModel(fixture.config);
  const initial = new Map();
  const { lossAbsolute: lossTolerance, weightAbsolute: weightTolerance } = fixture.referenceContract.tolerance;
  const steps = [];
  try {
    for (const [name, expected] of Object.entries(fixture.initialSha256)) {
      const values = await model.readParameter(name);
      const hash = new Uint8Array(await crypto.subtle.digest('SHA-256', values));
      const actual = Array.from(hash, x => x.toString(16).padStart(2, '0')).join('');
      if (actual !== expected) throw new Error('Tiny oracle initialization differs at ' + name);
      initial.set(name, values);
    }
    for (const entry of fixture.steps) {
      const tokens = Uint32Array.from(entry.sourceTokens);
      const loss = await model.trainStep(tokens.slice(0, -1), tokens.slice(1));
      const checked = new Map();
      let maxWeightError = 0;
      let largestExpectedUpdate = 0;
      let worstTensor = null;
      for (const { name, index, weight } of entry.expected) {
        if (!checked.has(name)) checked.set(name, await model.readParameter(name));
        const error = Math.abs(checked.get(name)[index] - weight);
        if (!Number.isFinite(error)) throw new Error('Nonfinite tiny oracle weight in ' + name);
        if (error > maxWeightError) {
          maxWeightError = error;
          worstTensor = name + '[' + index + ']';
        }
        largestExpectedUpdate = Math.max(largestExpectedUpdate, Math.abs(weight - initial.get(name)[index]));
      }
      steps.push({
        loss, referenceLoss: entry.loss, maxWeightError, worstTensor, largestExpectedUpdate,
        checkedCoordinates: entry.expected.length,
        pass: Math.abs(loss - entry.loss) <= lossTolerance && maxWeightError <= weightTolerance,
      });
    }
    const evalTokens = Uint32Array.from(fixture.heldOutTokens);
    const evalLoss = await model.evaluate(evalTokens.slice(0, -1), evalTokens.slice(1));
    return {
      reference: fixture.reference, torchVersion: fixture.torchVersion, model: fixture.config,
      initTensorDigestsChecked: initial.size, lossTolerance, weightTolerance, steps,
      evalLoss, referenceEvalLoss: fixture.heldOutLoss,
      pass: steps.every(step => step.pass) && Math.abs(evalLoss - fixture.heldOutLoss) <= lossTolerance,
    };
  } finally {
    model.destroy?.();
  }
}
