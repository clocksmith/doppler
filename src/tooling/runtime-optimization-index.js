import { computeCanonicalSha256 } from '../utils/canonical-hash.js';

export const RUNTIME_OPTIMIZATION_RESULTS_INDEX_SCHEMA =
  'doppler.runtime-optimization-results-index/v1';

function receiptIdentity(receipt) {
  const device = receipt.measurement?.pairs
    ?.find((pair) => pair.valid)
    ?.candidate?.deviceInfo ?? null;
  return {
    candidateKind: receipt.candidateKind ?? 'runtime-profile',
    candidateReferenceDigest: receipt.registeredReference?.digest ?? receipt.candidateHash,
    changeClass: receipt.campaign?.changeClass ?? null,
    modelId: receipt.model?.modelId ?? null,
    metricPath: receipt.measurement?.metricPath ?? null,
    deviceDigest: device ? computeCanonicalSha256(device) : null,
    workloadDigest: computeCanonicalSha256({
      runtimeInputs: receipt.runtimeInputs,
      metricPath: receipt.measurement?.metricPath ?? null,
    }),
  };
}

export function buildRuntimeOptimizationResultsIndex(receipts) {
  if (!Array.isArray(receipts)) {
    throw new Error('runtime optimization index: receipts must be an array');
  }
  const groups = new Map();
  for (const receipt of receipts) {
    if (receipt?.schema !== 'doppler.runtime-optimization-receipt/v1') {
      throw new Error('runtime optimization index: unsupported receipt schema');
    }
    const identity = receiptIdentity(receipt);
    const key = computeCanonicalSha256(identity);
    const group = groups.get(key) ?? {
      key,
      identity,
      accepted: 0,
      rejected: 0,
      invalid: 0,
      reasons: {},
      retryConditions: {},
      revocationConditions: {},
      receiptHashes: [],
    };
    const status = receipt.decision?.status;
    if (status === 'accepted') group.accepted += 1;
    else if (status === 'invalid') group.invalid += 1;
    else group.rejected += 1;
    for (const reason of receipt.decision?.reasons ?? []) {
      group.reasons[reason] = (group.reasons[reason] ?? 0) + 1;
    }
    for (const condition of receipt.campaign?.retryConditions ?? []) {
      group.retryConditions[condition] = (group.retryConditions[condition] ?? 0) + 1;
    }
    for (const condition of receipt.campaign?.revocationConditions ?? []) {
      group.revocationConditions[condition] = (group.revocationConditions[condition] ?? 0) + 1;
    }
    group.receiptHashes.push(receipt.receiptHash ?? computeCanonicalSha256(receipt));
    groups.set(key, group);
  }
  const entries = Array.from(groups.values())
    .map((entry) => ({
      ...entry,
      reasons: Object.fromEntries(
        Object.entries(entry.reasons).sort(([left], [right]) => left.localeCompare(right))
      ),
      retryConditions: Object.fromEntries(
        Object.entries(entry.retryConditions).sort(([left], [right]) => left.localeCompare(right))
      ),
      revocationConditions: Object.fromEntries(
        Object.entries(entry.revocationConditions).sort(([left], [right]) => left.localeCompare(right))
      ),
      receiptHashes: [...entry.receiptHashes].sort(),
    }))
    .sort((left, right) => left.key.localeCompare(right.key));
  const core = {
    schema: RUNTIME_OPTIMIZATION_RESULTS_INDEX_SCHEMA,
    receiptCount: receipts.length,
    negativeResultCount: entries.reduce(
      (count, entry) => count + entry.rejected + entry.invalid,
      0
    ),
    entries,
  };
  return { ...core, digest: computeCanonicalSha256(core) };
}
