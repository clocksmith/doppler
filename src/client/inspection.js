import observationPolicyRegistry from '../config/inspection/observation-policies.json' with { type: 'json' };
import { computeCanonicalSha256 } from '../utils/canonical-hash.js';

export const OBSERVATION_POLICY_REGISTRY_SCHEMA = 'doppler.observation-policy-registry/v1';
export const COMPARISON_FINGERPRINT_SCHEMA = 'doppler.comparison-fingerprint/v1';
export const MODEL_INSPECTION_RECEIPT_SCHEMA = 'doppler.model-inspection-receipt/v1';
export const WORD_SEGMENTATION_SCHEMA = 'doppler.word-segmentation/unicode-whitespace-v1';
export const PERPLEXITY_AGGREGATION_SCHEMA = 'doppler.perplexity/summed-word-surprisal-v1';

const COMPARISON_KINDS = new Set(['quality', 'performance']);
const policyById = new Map(
  observationPolicyRegistry.policies.map((policy) => [policy.id, Object.freeze(policy)])
);

function assertPlainObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${label} must be an object.`);
  }
}

function assertString(value, label) {
  const normalized = String(value ?? '').trim();
  if (!normalized) {
    throw new Error(`${label} must be a non-empty string.`);
  }
  return normalized;
}

function normalizeSha256(value, label) {
  const normalized = assertString(value, label).toLowerCase();
  const digest = normalized.startsWith('sha256:') ? normalized : `sha256:${normalized}`;
  if (!/^sha256:[0-9a-f]{64}$/.test(digest)) {
    throw new Error(`${label} must be a SHA-256 digest.`);
  }
  return digest;
}

function normalizeIntegerIds(values, label) {
  if (!Array.isArray(values) && !ArrayBuffer.isView(values)) {
    throw new Error(`${label} must be an array of token IDs.`);
  }
  const ids = Array.from(values, Number);
  if (ids.some((value) => !Number.isInteger(value) || value < 0)) {
    throw new Error(`${label} must contain non-negative integers.`);
  }
  return ids;
}

function cloneJson(value) {
  return value == null ? value : JSON.parse(JSON.stringify(value));
}

function normalizeBrowserIdentity(value = {}) {
  return {
    userAgent: String(value.userAgent ?? ''),
    platform: String(value.platform ?? ''),
    language: String(value.language ?? ''),
  };
}

function normalizeAdapterIdentity(value = {}) {
  return {
    vendor: value.vendor == null ? null : String(value.vendor),
    architecture: value.architecture == null ? null : String(value.architecture),
    device: value.device == null ? null : String(value.device),
    description: value.description == null ? null : String(value.description),
  };
}

function normalizeExecutionIdentity(value = {}) {
  return {
    backend: value.backend == null ? null : String(value.backend),
    executionPlanId: value.executionPlanId == null ? null : String(value.executionPlanId),
    kernelPathId: value.kernelPathId == null ? null : String(value.kernelPathId),
    kernelPathSource: value.kernelPathSource == null ? null : String(value.kernelPathSource),
    activationDtype: value.activationDtype == null ? null : String(value.activationDtype),
    hasF16: value.hasF16 === true,
    hasSubgroups: value.hasSubgroups === true,
  };
}

function buildQualityIdentity(identity) {
  return {
    artifact: identity.artifact,
    tokenizer: identity.tokenizer,
    promptTokenIds: identity.promptTokenIds,
    sampling: identity.sampling,
    observationPolicy: identity.observationPolicy,
    perplexity: identity.perplexity,
  };
}

function buildPerformanceIdentity(identity) {
  return {
    artifact: identity.artifact,
    tokenizer: identity.tokenizer,
    promptTokenIds: identity.promptTokenIds,
    sampling: identity.sampling,
    observationPolicy: identity.observationPolicy,
    execution: identity.execution,
    browser: identity.browser,
    adapter: identity.adapter,
  };
}

export function listObservationPolicies() {
  return observationPolicyRegistry.policies.map((policy) => cloneJson(policy));
}

export function resolveObservationPolicy(policyId = observationPolicyRegistry.defaultPolicyId) {
  const normalized = assertString(policyId, 'observation policy id');
  const policy = policyById.get(normalized);
  if (!policy) {
    throw new Error(`Unknown observation policy "${normalized}".`);
  }
  if (policy.performanceRepresentative && policy.modifiesExecution) {
    throw new Error(`${normalized}: a policy that modifies execution cannot represent performance.`);
  }
  if (policy.performanceRepresentative && policy.gpuTimestampQueries) {
    throw new Error(`${normalized}: representative coarse timing cannot enable GPU timestamp queries.`);
  }
  return cloneJson(policy);
}

export function buildComparisonFingerprint(input) {
  assertPlainObject(input, 'comparison fingerprint input');
  const policy = resolveObservationPolicy(input.observationPolicyId);
  const promptTokenIds = normalizeIntegerIds(input.promptTokenIds, 'promptTokenIds');
  const artifact = {
    modelId: assertString(input.artifact?.modelId, 'artifact.modelId'),
    manifestHash: normalizeSha256(input.artifact?.manifestHash, 'artifact.manifestHash'),
  };
  const tokenizerContract = cloneJson(input.tokenizer ?? null);
  if (!tokenizerContract) {
    throw new Error('comparison fingerprint requires tokenizer identity.');
  }
  const tokenizer = {
    contract: tokenizerContract,
    digest: computeCanonicalSha256(tokenizerContract),
  };
  const identity = {
    artifact,
    tokenizer,
    promptTokenIds,
    sampling: cloneJson(input.sampling ?? {}),
    observationPolicy: {
      id: policy.id,
      modifiesExecution: policy.modifiesExecution,
      performanceRepresentative: policy.performanceRepresentative,
      requiredCaptures: [...policy.requiredCaptures],
      allowedClaimTypes: [...policy.allowedClaimTypes],
    },
    perplexity: cloneJson(policy.perplexity),
    execution: normalizeExecutionIdentity(input.execution),
    browser: normalizeBrowserIdentity(input.browser),
    adapter: normalizeAdapterIdentity(input.adapter),
  };
  return {
    schema: COMPARISON_FINGERPRINT_SCHEMA,
    identity,
    fullDigest: computeCanonicalSha256(identity),
    qualityDigest: computeCanonicalSha256(buildQualityIdentity(identity)),
    performanceDigest: computeCanonicalSha256(buildPerformanceIdentity(identity)),
  };
}

export function assertComparableFingerprints(kind, left, right) {
  if (!COMPARISON_KINDS.has(kind)) {
    throw new Error(`Comparison kind must be "quality" or "performance"; received "${kind}".`);
  }
  if (left?.schema !== COMPARISON_FINGERPRINT_SCHEMA || right?.schema !== COMPARISON_FINGERPRINT_SCHEMA) {
    throw new Error('Both values must be Doppler comparison fingerprints.');
  }
  if (kind === 'quality') {
    if (left.identity?.tokenizer?.digest !== right.identity?.tokenizer?.digest) {
      throw new Error('Quality comparison rejected: tokenizer identities differ.');
    }
    if (left.qualityDigest !== right.qualityDigest) {
      throw new Error('Quality comparison rejected: canonical quality fingerprints differ.');
    }
  } else {
    if (left.performanceDigest !== right.performanceDigest) {
      throw new Error('Performance comparison rejected: canonical performance fingerprints differ.');
    }
    if (left.identity?.observationPolicy?.performanceRepresentative !== true) {
      throw new Error('Performance comparison rejected: observation policy is not representative.');
    }
    if (right.identity?.observationPolicy?.performanceRepresentative !== true) {
      throw new Error('Performance comparison rejected: observation policy is not representative.');
    }
  }
  return true;
}

function selectedProbability(logits, tokenId) {
  if (!logits || !Number.isInteger(tokenId) || tokenId < 0 || tokenId >= logits.length) {
    return null;
  }
  let maximum = -Infinity;
  for (let index = 0; index < logits.length; index += 1) {
    const value = Number(logits[index]);
    if (Number.isFinite(value) && value > maximum) maximum = value;
  }
  if (!Number.isFinite(maximum)) return null;
  let denominator = 0;
  for (let index = 0; index < logits.length; index += 1) {
    denominator += Math.exp(Number(logits[index]) - maximum);
  }
  if (!Number.isFinite(denominator) || denominator <= 0) return null;
  return Math.exp(Number(logits[tokenId]) - maximum) / denominator;
}

function topCandidates(logits, tokenizer, count) {
  if (!logits || !Number.isInteger(count) || count <= 0) return [];
  const candidates = [];
  for (let tokenId = 0; tokenId < logits.length; tokenId += 1) {
    const logit = Number(logits[tokenId]);
    if (!Number.isFinite(logit)) continue;
    if (candidates.length < count) {
      candidates.push({ tokenId, logit });
      candidates.sort((left, right) => right.logit - left.logit);
      continue;
    }
    if (logit > candidates[candidates.length - 1].logit) {
      candidates[candidates.length - 1] = { tokenId, logit };
      candidates.sort((left, right) => right.logit - left.logit);
    }
  }
  const maximum = candidates[0]?.logit ?? -Infinity;
  let denominator = 0;
  for (let index = 0; index < logits.length; index += 1) {
    denominator += Math.exp(Number(logits[index]) - maximum);
  }
  return candidates.map((candidate) => ({
    ...candidate,
    text: String(tokenizer.decode([candidate.tokenId], true, false)),
    probability: denominator > 0 ? Math.exp(candidate.logit - maximum) / denominator : null,
  }));
}

function segmentTokenRecords(tokenRecords) {
  const words = [];
  let current = null;
  const flush = () => {
    if (!current) return;
    current.wordIndex = words.length;
    words.push(current);
    current = null;
  };
  for (const token of tokenRecords) {
    const pieces = String(token.text ?? '').split(/(\s+)/u).filter(Boolean);
    if (pieces.length === 0) {
      pieces.push('');
    }
    for (const piece of pieces) {
      if (/^\s+$/u.test(piece)) {
        flush();
        continue;
      }
      if (!current) {
        current = {
          text: '',
          tokenIndexes: [],
          tokenCount: 0,
          summedSurprisal: 0,
          probabilityAvailable: true,
        };
      }
      current.text += piece;
      if (!current.tokenIndexes.includes(token.index)) {
        current.tokenIndexes.push(token.index);
        current.tokenCount += 1;
        if (Number.isFinite(token.surprisal)) {
          current.summedSurprisal += token.surprisal;
        } else {
          current.probabilityAvailable = false;
        }
      }
    }
  }
  flush();
  return words;
}

export function aggregateWordPerplexity(tokenRecords, options = {}) {
  const windowUnit = options.windowUnit ?? 'words';
  const windowSize = Number(options.windowSize ?? 8);
  if (windowUnit !== 'words' && windowUnit !== 'tokens') {
    throw new Error('Perplexity rolling window unit must be "words" or "tokens".');
  }
  if (!Number.isInteger(windowSize) || windowSize <= 0) {
    throw new Error('Perplexity rolling window size must be a positive integer.');
  }
  const words = segmentTokenRecords(tokenRecords);
  let cumulativeSurprisal = 0;
  let cumulativeTokens = 0;
  for (let index = 0; index < words.length; index += 1) {
    const word = words[index];
    if (word.probabilityAvailable) {
      cumulativeSurprisal += word.summedSurprisal;
      cumulativeTokens += word.tokenCount;
    }
    let selected = [];
    if (windowUnit === 'words') {
      selected = words.slice(Math.max(0, index - windowSize + 1), index + 1);
    } else {
      let remaining = windowSize;
      for (let cursor = index; cursor >= 0 && remaining > 0; cursor -= 1) {
        selected.unshift(words[cursor]);
        remaining -= words[cursor].tokenCount;
      }
    }
    const available = selected.filter((entry) => entry.probabilityAvailable);
    const rollingTokens = available.reduce((sum, entry) => sum + entry.tokenCount, 0);
    const rollingSurprisal = available.reduce((sum, entry) => sum + entry.summedSurprisal, 0);
    word.rollingPerplexity = rollingTokens > 0 ? Math.exp(rollingSurprisal / rollingTokens) : null;
    word.cumulativePerplexity = cumulativeTokens > 0
      ? Math.exp(cumulativeSurprisal / cumulativeTokens)
      : null;
    word.rollingWindow = { unit: windowUnit, size: windowSize, tokenCount: rollingTokens };
  }
  return {
    wordSegmentation: WORD_SEGMENTATION_SCHEMA,
    aggregation: PERPLEXITY_AGGREGATION_SCHEMA,
    rollingWindow: { unit: windowUnit, size: windowSize },
    words,
  };
}

export function buildInspectionTokenRecords(tokenIds, logitsByStep, tokenizer, topKSize = 5) {
  return tokenIds.map((tokenId, index) => {
    const logits = logitsByStep[index] ?? null;
    const probability = selectedProbability(logits, tokenId);
    return {
      index,
      tokenId,
      text: String(tokenizer.decode([tokenId], true, false)),
      probability,
      surprisal: probability && probability > 0 ? -Math.log(probability) : null,
      topCandidates: topCandidates(logits, tokenizer, topKSize),
    };
  });
}
