import { sha256Hex } from '../utils/sha256.js';
import { stableSortObject } from '../utils/stable-sort-object.js';

export const RELEASE_TO_JAVASCRIPT_RECEIPT_SCHEMA_ID = 'doppler.release-to-javascript-receipt/v1';

const DIGEST_PATTERN = /^sha256:[0-9a-f]{64}$/;

function isObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function hashValue(value) {
  return `sha256:${sha256Hex(JSON.stringify(stableSortObject(value)))}`;
}

function timestamp(value, label, errors) {
  if (typeof value !== 'string' || !Number.isFinite(Date.parse(value))) {
    errors.push(`${label} must be an ISO timestamp.`);
    return null;
  }
  return Date.parse(value);
}

function requireString(value, label, errors) {
  if (typeof value !== 'string' || !value.trim()) errors.push(`${label} must be a non-empty string.`);
}

function receiptCore(receipt) {
  const { receiptDigest: ignored, ...core } = receipt;
  void ignored;
  return core;
}

export function validateReleaseToJavaScriptReceipt(receipt) {
  const errors = [];
  if (!isObject(receipt)) return { ok: false, errors: ['Release-to-JavaScript receipt must be an object.'] };
  if (receipt.schema !== RELEASE_TO_JAVASCRIPT_RECEIPT_SCHEMA_ID) {
    errors.push(`schema must be "${RELEASE_TO_JAVASCRIPT_RECEIPT_SCHEMA_ID}".`);
  }
  requireString(receipt.campaignId, 'campaignId', errors);
  if (!isObject(receipt.source)) {
    errors.push('source must be an object.');
  } else {
    requireString(receipt.source.checkpointId, 'source.checkpointId', errors);
    requireString(receipt.source.revision, 'source.revision', errors);
  }
  const publishedAt = timestamp(receipt.source?.publishedAt, 'source.publishedAt', errors);
  const startedAt = timestamp(receipt.startedAt, 'startedAt', errors);
  const completedAt = timestamp(receipt.completedAt, 'completedAt', errors);
  if (!isObject(receipt.elapsed)) {
    errors.push('elapsed must be an object.');
  } else if (publishedAt != null && startedAt != null && completedAt != null) {
    if (receipt.elapsed.publicationToSignedPackMs !== completedAt - publishedAt) {
      errors.push('elapsed.publicationToSignedPackMs does not match source publication and completion timestamps.');
    }
    if (receipt.elapsed.forgeCampaignMs !== completedAt - startedAt) {
      errors.push('elapsed.forgeCampaignMs does not match campaign timestamps.');
    }
    if (completedAt < startedAt || startedAt < publishedAt) {
      errors.push('timestamps must satisfy publishedAt <= startedAt <= completedAt.');
    }
  }
  if (!Array.isArray(receipt.humanInterventions)) {
    errors.push('humanInterventions must be an array.');
  } else {
    for (const [index, intervention] of receipt.humanInterventions.entries()) {
      if (!isObject(intervention)) {
        errors.push(`humanInterventions[${index}] must be an object.`);
        continue;
      }
      requireString(intervention.id, `humanInterventions[${index}].id`, errors);
      requireString(intervention.kind, `humanInterventions[${index}].kind`, errors);
      requireString(intervention.actor, `humanInterventions[${index}].actor`, errors);
      requireString(intervention.disposition, `humanInterventions[${index}].disposition`, errors);
    }
    const semanticDecisionCount = receipt.humanInterventions.filter((entry) => (
      entry?.kind === 'semantic-decision' && entry?.disposition === 'accepted'
    )).length;
    if (receipt.humanAuthoredSemanticDecisions !== semanticDecisionCount) {
      errors.push('humanAuthoredSemanticDecisions must equal accepted human semantic-decision interventions.');
    }
  }
  if (!Array.isArray(receipt.unresolvedFacts)) errors.push('unresolvedFacts must be an array.');
  if (!isObject(receipt.candidates)) {
    errors.push('candidates must be an object.');
  } else {
    const { generated, rejected, accepted } = receipt.candidates;
    if (!Number.isInteger(generated) || generated < 1) errors.push('candidates.generated must be a positive integer.');
    if (!Array.isArray(rejected)) errors.push('candidates.rejected must be an array.');
    if (!Array.isArray(accepted) || accepted.length < 1) errors.push('candidates.accepted must be a non-empty array.');
    if (Array.isArray(rejected) && Array.isArray(accepted) && generated !== rejected.length + accepted.length) {
      errors.push('candidates.generated must equal rejected plus accepted candidates.');
    }
  }
  if (!isObject(receipt.acceptedCode)) {
    errors.push('acceptedCode must be an object.');
  } else {
    requireString(receipt.acceptedCode.revision, 'acceptedCode.revision', errors);
    if (!Array.isArray(receipt.acceptedCode.files) || receipt.acceptedCode.files.length < 1
      || receipt.acceptedCode.files.some((file) => typeof file !== 'string' || !file.trim())) {
      errors.push('acceptedCode.files must be a non-empty string array.');
    }
    if (!DIGEST_PATTERN.test(receipt.acceptedCode.digest || '')) {
      errors.push('acceptedCode.digest must be a SHA-256 digest.');
    }
  }
  if (!isObject(receipt.qualification) || receipt.qualification.status !== 'passed') {
    errors.push('qualification.status must be "passed".');
  } else {
    requireString(receipt.qualification.packId, 'qualification.packId', errors);
    if (!DIGEST_PATTERN.test(receipt.qualification.packDigest || '')) {
      errors.push('qualification.packDigest must be a SHA-256 digest.');
    }
  }
  if (!Array.isArray(receipt.evidence) || receipt.evidence.length < 1) {
    errors.push('evidence must be a non-empty array.');
  } else {
    for (const [index, entry] of receipt.evidence.entries()) {
      requireString(entry?.kind, `evidence[${index}].kind`, errors);
      requireString(entry?.path, `evidence[${index}].path`, errors);
      if (!DIGEST_PATTERN.test(entry?.digest || '')) errors.push(`evidence[${index}].digest must be a SHA-256 digest.`);
    }
  }
  if (!DIGEST_PATTERN.test(receipt.receiptDigest || '')) {
    errors.push('receiptDigest must be a SHA-256 digest.');
  } else if (receipt.receiptDigest !== hashValue(receiptCore(receipt))) {
    errors.push('receiptDigest mismatch.');
  }
  return { ok: errors.length === 0, errors };
}

export function createReleaseToJavaScriptReceipt(fields) {
  const publishedAt = Date.parse(fields?.source?.publishedAt);
  const startedAt = Date.parse(fields?.startedAt);
  const completedAt = Date.parse(fields?.completedAt);
  const humanInterventions = structuredClone(fields?.humanInterventions ?? []);
  const core = {
    schema: RELEASE_TO_JAVASCRIPT_RECEIPT_SCHEMA_ID,
    campaignId: fields?.campaignId,
    source: structuredClone(fields?.source),
    startedAt: fields?.startedAt,
    completedAt: fields?.completedAt,
    elapsed: {
      publicationToSignedPackMs: completedAt - publishedAt,
      forgeCampaignMs: completedAt - startedAt,
    },
    humanInterventions,
    humanAuthoredSemanticDecisions: humanInterventions.filter((entry) => (
      entry?.kind === 'semantic-decision' && entry?.disposition === 'accepted'
    )).length,
    unresolvedFacts: structuredClone(fields?.unresolvedFacts ?? []),
    candidates: structuredClone(fields?.candidates),
    acceptedCode: structuredClone(fields?.acceptedCode),
    qualification: structuredClone(fields?.qualification),
    evidence: structuredClone(fields?.evidence),
  };
  const receipt = { ...core, receiptDigest: hashValue(core) };
  const validation = validateReleaseToJavaScriptReceipt(receipt);
  if (!validation.ok) throw new Error(`Invalid Release-to-JavaScript receipt: ${validation.errors.join('; ')}`);
  return Object.freeze(receipt);
}
