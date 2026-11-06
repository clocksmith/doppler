import assert from 'node:assert/strict';

// Observation only: these errors do not define a post-hoc acceptance budget.
export function compareLearnedLayerCapture(capture, expected, positions, stride) {
  assert.ok(capture && !capture.error && Array.isArray(capture.data), 'complete layer capture required');
  const [tokens, width] = capture.shape;
  assert.equal(capture.data.length, tokens * width);
  assert.ok(capture.data.every(Number.isFinite), 'layer activations must be finite');
  assert.ok(Number.isSafeInteger(stride) && stride > 0);
  assert.equal(expected.samples.length, positions.length);
  let squaredError = 0, squaredReference = 0, maxAbsoluteError = 0, count = 0;
  const rows = positions.map((position, index) => {
    assert.ok(Number.isSafeInteger(position) && position >= 0 && position < tokens);
    const reference = expected.samples[index];
    assert.equal(reference.length, Math.ceil(width / stride));
    const observed = reference.map((value, sample) => {
      assert.ok(Number.isFinite(value));
      const actual = capture.data[position * width + sample * stride];
      const delta = actual - value;
      squaredError += delta * delta;
      squaredReference += value * value;
      maxAbsoluteError = Math.max(maxAbsoluteError, Math.abs(delta));
      count++;
      return actual;
    });
    return { position, observed };
  });
  return { shape: capture.shape, fullTensorDigest: capture.fullTensorDigest, sampleCount: count,
    maxAbsoluteError, rmsError: Math.sqrt(squaredError / count),
    relativeL2: squaredReference > 0 ? Math.sqrt(squaredError / squaredReference) : null, rows };
}
