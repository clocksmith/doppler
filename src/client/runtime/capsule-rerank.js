import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';

export const CAPSULE_RERANK_RECEIPT_SCHEMA = 'doppler.capsule-rerank-receipt/v1';

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Capsule rerank requires ${label} as an object.`);
  }
}

function assertExactIdentity(actual, expected, label) {
  assertObject(actual, label);
  for (const field of Object.keys(expected)) {
    if (expected[field] && typeof expected[field] === 'object') {
      assertExactIdentity(actual[field], expected[field], `${label}.${field}`);
    } else if (actual[field] !== expected[field]) {
      throw new Error(`Capsule rerank ${label}.${field} does not match the signed Capsule release contract.`);
    }
  }
  const extra = Object.keys(actual).filter((field) => !(field in expected));
  if (extra.length > 0) {
    throw new Error(`Capsule rerank ${label} contains undeclared fields: ${extra.join(', ')}.`);
  }
}

function assertApplicationBinding(request, capsule) {
  assertObject(request, 'request');
  const expected = capsule.release.application;
  assertObject(request.application, 'request.application');
  assertExactIdentity(request.application, {
    applicationId: expected.applicationId,
    applicationRevision: expected.applicationRevision,
    applicationRevisionDigest: expected.applicationRevisionDigest,
    workload: expected.workload,
    oracle: expected.oracle,
  }, 'request.application');
  if (typeof request.query !== 'string' || !request.query.trim()) {
    throw new Error('Capsule rerank request.query must be a non-empty string.');
  }
  if (!Array.isArray(request.documents) || request.documents.length === 0) {
    throw new Error('Capsule rerank request.documents must be a non-empty array.');
  }
  for (const [index, document] of request.documents.entries()) {
    if (typeof document !== 'string' || !document.trim()) {
      throw new Error(`Capsule rerank request.documents[${index}] must be a non-empty string.`);
    }
  }
}

function assertModelEvidence(evidence) {
  if (evidence?.schema !== 'doppler_rerank_evidence/v1') {
    throw new Error('Capsule rerank program must return Doppler rerank evidence v1.');
  }
  for (const field of ['inputHash', 'outputHash', 'backendIdentityHash']) {
    if (!/^sha256:[0-9a-f]{64}$/.test(evidence[field] || '')) {
      throw new Error(`Capsule rerank evidence.${field} must be a SHA-256 digest.`);
    }
  }
}

export async function executeCapsuleRerank({
  capsule,
  targetPlan,
  targetPlanDigest,
  program,
  request,
}) {
  assertApplicationBinding(request, capsule);
  request.options?.signal?.throwIfAborted();
  if (typeof program?.rerank !== 'function') {
    throw new Error('Selected Capsule program does not implement its declared rerank workload.');
  }
  const evidence = await program.rerank({
    query: request.query,
    documents: request.documents,
    options: request.options,
  });
  request.options?.signal?.throwIfAborted();
  assertModelEvidence(evidence);
  const payload = {
    schema: CAPSULE_RERANK_RECEIPT_SCHEMA,
    capsule: {
      capsuleId: capsule.capsuleId,
      semanticRoot: capsule.semanticRoot,
      modelId: capsule.modelId,
      signingAuthority: capsule.signature.authority,
    },
    application: capsule.release.application,
    target: {
      targetId: targetPlan.targetId,
      targetPlanDigest,
    },
    lifecycle: {
      releaseVersion: capsule.release.lifecycle.releaseVersion,
      previousCapsuleId: capsule.release.lifecycle.failedUpgrade.previousCapsuleId,
      previousSemanticRoot: capsule.release.lifecycle.failedUpgrade.previousSemanticRoot,
    },
    revocation: capsule.release.revocation,
    evidence,
  };
  return Object.freeze({
    ...payload,
    receiptDigest: computeCanonicalSha256(payload),
  });
}
