import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';

// The frozen Qwen3.5 0.8B reference manifest, byte-identical to the copy Reploid
// added in a33cf898 (tests/fixtures/distributed-reference-manifest.json.gz).
const FROZEN_REFERENCE_IDENTITY = 'sha256:edeb69dd65cbb26971abf773a95bf6dc219a82942c0951d65e1893d442b607c9';

export function readFrozenReferenceManifest() {
  const bytes = gunzipSync(readFileSync(new URL('../fixtures/qwen35-frozen-reference-manifest.json.gz', import.meta.url)));
  assert.equal('sha256:' + createHash('sha256').update(bytes).digest('hex'), FROZEN_REFERENCE_IDENTITY);
  return JSON.parse(bytes);
}

// Everything except refresh time, conversion/capsule ids and kernel digests must match.
function compareReferenceModel(model, baseline) {
  const project = value => {
    const copy = structuredClone(value);
    delete copy.artifactIdentity.conversionConfigDigest;
    delete copy.artifactIdentity.weightCapsuleId;
    delete copy.metadata.manifestRefresh.at;
    for (const kernel of Object.values(copy.inference.execution.kernels)) delete kernel.digest;
    return copy;
  };
  if (model.artifactIdentity.weightCapsuleId !== undefined) {
    assert.equal(model.artifactIdentity.weightCapsuleId, baseline.artifactIdentity.weightPackId);
  }
  assert.deepEqual(project(model), project(baseline),
    'Frozen comparison must retain exact weights, tokenizer, layers, precision, cache and declared shader entries');
}

// The explicitly approved program/representation changes do not authorize a
// weight, tokenizer, source equation, source epsilon, or F16 KV policy change.
export function bindPairedModel(model, baseline, execution) {
  assert.deepEqual(model.inference.execution, execution, 'Paired graph differs from the frozen declared program.');
  const sourceFacts = structuredClone(model);
  sourceFacts.inference.execution = structuredClone(baseline.inference.execution);
  compareReferenceModel(sourceFacts, baseline);
  return { weightsTokenizerAndSourceFactsUnchanged: true, executionGraphChanged: true,
    precisionAuthorization: 'User-approved interleaved two-F32 intermediate and persistent state chain',
    publicLogits: 'f32', kvStorage: 'f16', numericalReference: 'unchanged approved independent F64', tolerance: 0.001 };
}
