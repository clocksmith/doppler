import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { freezeCapsuleV2 } from '../../config/capsule-v2.js';

export async function executeCapsuleForecast({ identity, release, targetPlan, targetPlanDigest, program, request, artifactReceipts, releaseEventDigest }) {
  const { signal, ...input } = request;
  const frozen = freezeCapsuleV2(structuredClone(input));
  if (!release?.application || computeCanonicalSha256(frozen.application) !== computeCanonicalSha256(release.application)) {
    throw new Error('Forecast application identity does not match its signed release contract.');
  }
  if (Object.keys(frozen).some(key => !['application', 'context', 'horizon', 'assignmentHash'].includes(key))
    || (frozen.assignmentHash !== null && !/^sha256:[0-9a-f]{64}$/.test(frozen.assignmentHash ?? ''))
    || !Array.isArray(frozen.context) || frozen.context.some(value => !Number.isFinite(value))
    || !Number.isSafeInteger(frozen.horizon) || frozen.horizon < 1
    || !targetPlan.qualification.some(record => record.operation === 'forecast') || typeof program?.forecast !== 'function') {
    throw new Error('Invalid or undeclared Capsule forecast request.');
  }
  if (signal?.aborted) throw signal.reason ?? new Error('Forecast cancelled.');
  const output = await program.forecast(frozen, { signal });
  if (signal?.aborted) throw signal.reason ?? new Error('Forecast cancelled.');
  if (output?.horizon !== frozen.horizon || output.layout !== 'time-quantile'
    || !Array.isArray(output.quantileLevels) || output.quantileLevels.length === 0
    || output.quantileLevels.some((q, i, levels) => !Number.isFinite(q) || q <= 0 || q >= 1 || (i > 0 && q <= levels[i - 1]))
    || !Array.isArray(output.values) || output.values.length !== output.horizon * output.quantileLevels.length
    || output.values.some(value => !Number.isFinite(value))) throw new Error('Malformed Capsule forecast output.');
  const payload = {
    schema: 'doppler.capsule-execution-receipt/v1', operation: 'forecast', capsule: identity,
    targetId: targetPlan.targetId, targetPlanDigest, artifactReceipts, releaseEventDigest,
    assignmentHash: frozen.assignmentHash, inputHash: computeCanonicalSha256(frozen), outputHash: computeCanonicalSha256(output),
  };
  return freezeCapsuleV2({ ...output, receipt: { ...payload, receiptDigest: computeCanonicalSha256(payload) } });
}
