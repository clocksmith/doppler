let activeManifest = null;
let lastRolloutDecision = null;

function normalizeRolloutPolicy(policy) {
  const rollout = policy?.rollout && typeof policy.rollout === 'object'
    ? policy.rollout
    : {};
  const rawMode = String(rollout.mode || 'shadow').trim().toLowerCase().replace(/_/g, '-');
  if (rawMode !== 'default' && rawMode !== 'canary' && rawMode !== 'opt-in' && rawMode !== 'shadow') {
    throw new Error(
      `hotswap.rollout.mode must be one of default, canary, opt-in, shadow (received "${rollout.mode}")`
    );
  }
  let canaryPercent = 0;
  if (rollout.canaryPercent !== undefined && rollout.canaryPercent !== null) {
    const parsedCanaryPercent = Number(rollout.canaryPercent);
    if (!Number.isFinite(parsedCanaryPercent) || parsedCanaryPercent < 0 || parsedCanaryPercent > 100) {
      throw new Error('hotswap.rollout.canaryPercent must be a number between 0 and 100 when provided.');
    }
    canaryPercent = parsedCanaryPercent;
  }
  if (rollout.cohortSalt !== undefined && rollout.cohortSalt !== null && typeof rollout.cohortSalt !== 'string') {
    throw new Error('hotswap.rollout.cohortSalt must be a string when provided.');
  }
  const cohortSalt = String(rollout.cohortSalt || 'doppler-hotswap-v1').trim() || 'doppler-hotswap-v1';
  if (rollout.optInAllowlist !== undefined && rollout.optInAllowlist !== null && !Array.isArray(rollout.optInAllowlist)) {
    throw new Error('hotswap.rollout.optInAllowlist must be an array of strings when provided.');
  }
  const optInAllowlist = Array.isArray(rollout.optInAllowlist)
    ? rollout.optInAllowlist.map((entry, index) => {
      if (typeof entry !== 'string') {
        throw new Error(`hotswap.rollout.optInAllowlist[${index}] must be a string.`);
      }
      const normalized = entry.trim();
      if (!normalized) {
        throw new Error(`hotswap.rollout.optInAllowlist[${index}] must not be empty.`);
      }
      return normalized;
    })
    : [];
  return {
    mode: rawMode,
    canaryPercent,
    cohortSalt,
    optInAllowlist,
  };
}

function hashBucket(value) {
  const source = String(value || '');
  let hash = 2166136261;
  for (let i = 0; i < source.length; i += 1) {
    hash ^= source.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return Math.abs(hash >>> 0) % 100;
}

export function evaluateHotSwapRollout(policy = {}, context = {}) {
  if (policy?.enabled !== true) {
    const disabled = {
      allowed: false,
      mode: 'disabled',
      reason: 'hotswap_disabled',
      bucket: null,
      threshold: null,
      subjectId: null,
    };
    lastRolloutDecision = disabled;
    return disabled;
  }

  const rollout = normalizeRolloutPolicy(policy);
  const subjectId = String(
    context.subjectId
    || context.modelId
    || context.modelUrl
    || context.sessionId
    || 'anonymous'
  );
  const forced = context.forceEnable === true;
  const optInTag = String(context.optInTag || '').trim();
  const bucket = hashBucket(`${rollout.cohortSalt}:${subjectId}`);

  let decision = null;
  if (forced) {
    decision = {
      allowed: true,
      mode: rollout.mode,
      reason: 'force_enable',
      bucket,
      threshold: rollout.mode === 'canary' ? rollout.canaryPercent : null,
      subjectId,
    };
  } else if (rollout.mode === 'default') {
    decision = {
      allowed: true,
      mode: rollout.mode,
      reason: 'default_enabled',
      bucket,
      threshold: null,
      subjectId,
    };
  } else if (rollout.mode === 'canary') {
    decision = {
      allowed: bucket < rollout.canaryPercent,
      mode: rollout.mode,
      reason: bucket < rollout.canaryPercent ? 'canary_selected' : 'canary_excluded',
      bucket,
      threshold: rollout.canaryPercent,
      subjectId,
    };
  } else if (rollout.mode === 'opt-in') {
    const allowlisted = rollout.optInAllowlist.includes(subjectId);
    const tagged = optInTag.length > 0 && rollout.optInAllowlist.includes(optInTag);
    const allowed = allowlisted === true || tagged === true;
    decision = {
      allowed,
      mode: rollout.mode,
      reason: allowed ? 'opt_in_selected' : 'opt_in_required',
      bucket,
      threshold: null,
      subjectId,
    };
  } else {
    decision = {
      allowed: false,
      mode: rollout.mode,
      reason: 'shadow_mode',
      bucket,
      threshold: null,
      subjectId,
    };
  }

  lastRolloutDecision = decision;
  return decision;
}

export function getHotSwapManifest() {
  return activeManifest;
}

export function setHotSwapManifest(manifest) {
  activeManifest = manifest;
}

export function getLastHotSwapRolloutDecision() {
  return lastRolloutDecision;
}
