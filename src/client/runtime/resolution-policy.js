const ARTIFACTS = 'allowedArtifactVariantIds';
const EXECUTIONS = 'allowedExecutionIds';
const POLICY_FIELDS = new Set([ARTIFACTS, EXECUTIONS]);

const UNRESTRICTED_POLICY = Object.freeze({
  allowedArtifactVariantIds: null,
  allowedExecutionIds: null,
});

function normalizeIdentity(value, field) {
  const text = String(value ?? '').trim().toLowerCase();
  const identity = text.startsWith('sha256:') ? text : `sha256:${text}`;
  if (!/^sha256:[0-9a-f]{64}$/.test(identity)) {
    throw new Error(`Doppler resolutionPolicy.${field} requires SHA-256 identities.`);
  }
  return identity;
}

function normalizeList(value, field) {
  if (value == null) return null;
  if (!Array.isArray(value)) {
    throw new Error(`Doppler resolutionPolicy.${field} must be an array or null.`);
  }
  return Object.freeze([...new Set(value.map((entry) => normalizeIdentity(entry, field)))]);
}

export function resolveResolutionPolicy(value) {
  if (value == null) return UNRESTRICTED_POLICY;
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error('Doppler resolutionPolicy must be an object or null.');
  }
  for (const field of Object.keys(value)) {
    if (!POLICY_FIELDS.has(field)) {
      throw new Error(`Unknown Doppler resolutionPolicy field: ${field}.`);
    }
  }
  return Object.freeze({
    [ARTIFACTS]: normalizeList(value[ARTIFACTS], ARTIFACTS),
    [EXECUTIONS]: normalizeList(value[EXECUTIONS], EXECUTIONS),
  });
}

function assertAllowed(allowlist, value, field, label) {
  if (allowlist === null) return;
  const identity = normalizeIdentity(value, field);
  if (!allowlist.includes(identity)) {
    throw new Error(
      `Doppler resolutionPolicy rejected ${label} ${identity}; no authorized alternative matched.`
    );
  }
}

export function assertArtifactVariantAllowed(policy, manifestHash) {
  assertAllowed(
    policy[ARTIFACTS],
    manifestHash,
    ARTIFACTS,
    'artifact variant'
  );
}

export function assertExecutionAllowed(policy, resolvedExecutionId) {
  assertAllowed(
    policy[EXECUTIONS],
    resolvedExecutionId,
    EXECUTIONS,
    'execution'
  );
}

export function assertExecutionMayStart(policy) {
  if (policy.allowedExecutionIds?.length === 0) {
    throw new Error('Doppler resolutionPolicy.allowedExecutionIds rejects every execution.');
  }
}

export function assertUnreceiptedExecutionAllowed(policy, apiName) {
  if (policy.allowedExecutionIds !== null) {
    throw new Error(
      `${apiName} cannot run with resolutionPolicy.allowedExecutionIds because it may expose ` +
      'output before the final execution identity is verified. Use an evidence-bearing method.'
    );
  }
}
