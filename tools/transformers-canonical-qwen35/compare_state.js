// Observation only. F32 subtraction and tolerance arithmetic match NumPy's
// existing F32 strict gate; no value is fed back into model execution.
export function compareState(actual, reference, policy) {
  if (!(actual instanceof Float32Array) || !(reference instanceof Float32Array) || actual.length !== reference.length) {
    throw new Error('State comparison requires equal-size physical F32 observations');
  }
  if (!actual.length || !Number.isFinite(policy.stateAbsoluteTolerance) || !Number.isFinite(policy.stateRelativeTolerance) ||
      policy.stateAbsoluteTolerance < 0 || policy.stateRelativeTolerance < 0) throw new Error('State comparison requires data and explicit finite tolerances');
  const absolute = Math.fround(policy.stateAbsoluteTolerance), relative = Math.fround(policy.stateRelativeTolerance);
  let maximum = 0, sum = 0, failed = 0, nonFinite = 0, firstFailedElement = null;
  for (let index = 0; index < actual.length; index++) {
    const error = Math.abs(Math.fround(actual[index] - reference[index]));
    const limit = Math.fround(absolute + Math.fround(relative * Math.abs(reference[index])));
    const finite = Number.isFinite(actual[index]) && Number.isFinite(reference[index]);
    if (!finite) nonFinite++;
    if (!finite || error > limit) {
      failed++;
      firstFailedElement ??= { index, actual: finite ? actual[index] : String(actual[index]),
        native: finite ? reference[index] : String(reference[index]), absoluteError: finite ? error : null, limit };
    }
    if (finite) { maximum = Math.max(maximum, error); sum += error; }
  }
  return { elements: actual.length, maxAbsoluteError: nonFinite ? null : maximum,
    meanAbsoluteError: nonFinite ? null : sum / actual.length, meanAccumulation: 'float64 observation only',
    failedElements: failed, nonFiniteElements: nonFinite, firstFailedElement, pass: failed === 0 };
}
