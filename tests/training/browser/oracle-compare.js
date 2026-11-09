// Shared comparison mechanics for the training browser oracles. Numerical
// reference implementations and tolerances stay with each oracle; the metric
// does not. Lengths must match: an actual array longer or shorter than its
// reference is a failure, never a partial pass over the shorter prefix.

function isNumericArray(value) {
  return ArrayBuffer.isView(value) ? !(value instanceof DataView) : Array.isArray(value);
}

export function compare(actual, expected) {
  if (!isNumericArray(actual) || !isNumericArray(expected)) {
    throw new Error('compare() requires two numeric arrays.');
  }
  if (actual.length !== expected.length) {
    throw new Error(`compare() length mismatch: actual has ${actual.length} elements, expected has ${expected.length}.`);
  }
  let maxAbsError = 0;
  let squaredError = 0;
  let actualFinite = true;
  let expectedFinite = true;
  for (let index = 0; index < expected.length; index += 1) {
    const error = Math.abs(actual[index] - expected[index]);
    maxAbsError = Math.max(maxAbsError, error);
    squaredError += error * error;
    actualFinite = actualFinite && Number.isFinite(actual[index]);
    expectedFinite = expectedFinite && Number.isFinite(expected[index]);
  }
  // A non-finite reference is as much a failure as a non-finite result.
  return {
    elementCount: expected.length,
    allFinite: actualFinite && expectedFinite,
    actualFinite,
    expectedFinite,
    maxAbsError,
    rmse: expected.length ? Math.sqrt(squaredError / expected.length) : 0,
  };
}

export function maxAbsDifference(left, right) {
  return compare(left, right).maxAbsError;
}
