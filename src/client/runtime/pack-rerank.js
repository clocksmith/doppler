import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';

export const PACK_RERANK_RECEIPT_SCHEMA = 'doppler.pack-rerank-receipt/v1';

function assertObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Pack rerank requires ${label} as an object.`);
  }
}

function assertExactIdentity(actual, expected, label) {
  assertObject(actual, label);
  for (const field of Object.keys(expected)) {
    if (expected[field] && typeof expected[field] === 'object') {
      assertExactIdentity(actual[field], expected[field], `${label}.${field}`);
    } else if (actual[field] !== expected[field]) {
      throw new Error(`Pack rerank ${label}.${field} does not match the signed Pack release contract.`);
    }
  }
  const extra = Object.keys(actual).filter((field) => !(field in expected));
  if (extra.length > 0) {
    throw new Error(`Pack rerank ${label} contains undeclared fields: ${extra.join(', ')}.`);
  }
}

function assertApplicationBinding(request, pack) {
  assertObject(request, 'request');
  const expected = pack.release.application;
  assertObject(request.application, 'request.application');
  assertExactIdentity(request.application, {
    applicationId: expected.applicationId,
    applicationRevision: expected.applicationRevision,
    applicationRevisionDigest: expected.applicationRevisionDigest,
    workload: expected.workload,
    oracle: expected.oracle,
  }, 'request.application');
  if (typeof request.query !== 'string' || !request.query.trim()) {
    throw new Error('Pack rerank request.query must be a non-empty string.');
  }
  if (!Array.isArray(request.documents) || request.documents.length === 0) {
    throw new Error('Pack rerank request.documents must be a non-empty array.');
  }
  for (const [index, document] of request.documents.entries()) {
    if (typeof document !== 'string' || !document.trim()) {
      throw new Error(`Pack rerank request.documents[${index}] must be a non-empty string.`);
    }
  }
}

function assertModelEvidence(evidence) {
  if (evidence?.schema !== 'doppler_rerank_evidence/v1') {
    throw new Error('Pack rerank program must return Doppler rerank evidence v1.');
  }
  for (const field of ['inputHash', 'outputHash', 'backendIdentityHash']) {
    if (!/^sha256:[0-9a-f]{64}$/.test(evidence[field] || '')) {
      throw new Error(`Pack rerank evidence.${field} must be a SHA-256 digest.`);
    }
  }
}

export async function executePackRerank({
  pack,
  targetPlan,
  targetPlanDigest,
  program,
  request,
}) {
  assertApplicationBinding(request, pack);
  if (typeof program?.rerank !== 'function') {
    throw new Error('Selected Pack program does not implement its declared rerank workload.');
  }
  const evidence = await program.rerank({
    query: request.query,
    documents: request.documents,
    options: request.options,
  });
  assertModelEvidence(evidence);
  const payload = {
    schema: PACK_RERANK_RECEIPT_SCHEMA,
    pack: {
      packId: pack.packId,
      semanticRoot: pack.semanticRoot,
      modelId: pack.modelId,
      signingAuthority: pack.signature.authority,
    },
    application: pack.release.application,
    target: {
      targetId: targetPlan.targetId,
      targetPlanDigest,
    },
    lifecycle: {
      releaseVersion: pack.release.lifecycle.releaseVersion,
      previousPackId: pack.release.lifecycle.failedUpgrade.previousPackId,
      previousSemanticRoot: pack.release.lifecycle.failedUpgrade.previousSemanticRoot,
    },
    revocation: pack.release.revocation,
    evidence,
  };
  return Object.freeze({
    ...payload,
    receiptDigest: computeCanonicalSha256(payload),
  });
}
