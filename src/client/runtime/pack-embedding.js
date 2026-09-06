import { computeCanonicalSha256 } from '../../formats/canonical-hash.js';
import { freezePackV2 } from '../../config/pack-v2.js';
import { resolvePackEmbeddingContract } from '../../config/embedding-contract.js';

function requireObject(value, label) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`Pack embed requires ${label} as an object.`);
  }
}

export async function executePackEmbedding({
  identity, release, manifest, manifestHash, targetPlan, targetPlanDigest,
  program, request, artifactReceipts, releaseEventDigest,
}) {
  requireObject(request, 'request');
  requireObject(request.application, 'request.application');
  if (Object.keys(request).some(key => !['application', 'text', 'options'].includes(key))) {
    throw new Error('Pack embed request contains undeclared fields.');
  }
  if (request.options !== undefined) {
    requireObject(request.options, 'request.options');
    if (Object.keys(request.options).some(key => key !== 'signal')) {
      throw new Error('Pack embed options may contain only signal; model semantics belong to the signed manifest.');
    }
  }
  const { signal } = request.options ?? {};
  if (signal !== undefined && (typeof signal?.throwIfAborted !== 'function'
    || typeof signal.addEventListener !== 'function' || typeof signal.aborted !== 'boolean')) {
    throw new Error('Pack embed options.signal must be an AbortSignal.');
  }
  const input = freezePackV2(structuredClone({ application: request.application, text: request.text }));
  if (!release?.application || computeCanonicalSha256(input.application) !== computeCanonicalSha256(release.application)) {
    throw new Error('Pack embed application identity does not match its signed release contract.');
  }
  if (typeof input.text !== 'string' || !input.text.trim()) {
    throw new Error('Pack embed request.text must be a non-empty string.');
  }
  const contract = resolvePackEmbeddingContract(manifest);
  if (typeof program?.embed !== 'function') {
    throw new Error('Selected Pack program does not implement text embedding.');
  }
  signal?.throwIfAborted();
  const evidence = await program.embed(input.text, { signal });
  signal?.throwIfAborted();
  if (evidence?.schema !== 'doppler_embedding_evidence/v1'
    || !(Array.isArray(evidence.embedding) || evidence.embedding instanceof Float32Array)
    || evidence.embedding.length !== contract.dimension
    || Array.from(evidence.embedding).some(value => !Number.isFinite(value))
    || !Array.isArray(evidence.tokens) || evidence.tokens.length === 0
    || Array.from(evidence.tokens).some(value => !Number.isSafeInteger(value) || value < 0)
    || evidence.seqLen !== evidence.tokens.length || evidence.embeddingMode !== contract.postprocessor.poolingMode) {
    throw new Error('Malformed Pack text embedding output or undeclared output geometry.');
  }
  const output = { embedding: Array.from(evidence.embedding), tokens: [...evidence.tokens],
    seqLen: evidence.seqLen, embeddingMode: evidence.embeddingMode };
  if (evidence.resolution?.schema !== 'doppler.resolution-identity/v1'
    || evidence.executionIdentity?.schema !== 'doppler.resolved-execution-identity/v1'
    || evidence.backendIdentity?.backend !== 'webgpu'
    || evidence.inputHash !== computeCanonicalSha256({ text: input.text })
    || evidence.outputHash !== computeCanonicalSha256(output)
    || evidence.resolution?.resolvedArtifactVariantId !== manifestHash
    || evidence.resolution?.resolvedExecutionId !== computeCanonicalSha256(evidence.executionIdentity)
    || evidence.backendIdentityHash !== computeCanonicalSha256(evidence.backendIdentity)
    || computeCanonicalSha256(evidence.executionIdentity?.backendIdentity) !== evidence.backendIdentityHash) {
    throw new Error('Pack embed evidence does not match its input, output, manifest, or execution identity.');
  }
  const payload = {
    schema: 'doppler.pack-execution-receipt/v1', operation: 'embed', pack: identity,
    targetId: targetPlan.targetId, targetPlanDigest, artifactReceipts, releaseEventDigest,
    assignmentHash: null, inputHash: computeCanonicalSha256(input), outputHash: evidence.outputHash,
    application: input.application, resolution: evidence.resolution,
    executionIdentity: evidence.executionIdentity, backendIdentityHash: evidence.backendIdentityHash,
  };
  return freezePackV2(structuredClone({ ...output,
    receipt: { ...payload, receiptDigest: computeCanonicalSha256(payload) } }));
}
