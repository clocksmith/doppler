import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { snapshotFromArray } from '../../src/debug/tensor.js';
import {
  buildRuntimeBoundaryCapture,
  buildSourceBoundaryPack,
  buildSourceBoundaryPackFromProviderCapture,
  buildDeterministicTokenEvidenceFromReferenceTranscript,
  compareBoundaryEvidence,
} from '../../src/tooling/boundary-evidence.js';
import { computeCanonicalSha256 } from '../../src/utils/canonical-hash.js';

const repoRoot = path.resolve(new URL('../..', import.meta.url).pathname);
const policy = JSON.parse(
  await fs.readFile(
    path.join(repoRoot, 'src/config/evidence/boundary-evidence-policy.json'),
    'utf8'
  )
);
const qPre = snapshotFromArray(new Float32Array([1, 2, 3, 4]), [1, 4]);
const qPost = snapshotFromArray(new Float32Array([1.5, 2.5, 3.5, 4.5]), [1, 4]);
const report = {
  result: {
    metrics: {
      operatorDiagnostics: {
        timeline: [
          {
            stageName: 'attn.q_proj',
            layerIndex: 0,
            phase: 'prefill',
            tokenIndex: 0,
            capture: qPre,
          },
          {
            stageName: 'attn.q_rope',
            layerIndex: 0,
            phase: 'prefill',
            tokenIndex: 0,
            capture: qPost,
          },
        ],
      },
    },
  },
};

const runtimeCapture = buildRuntimeBoundaryCapture({ report, policy });
assert.deepEqual(
  runtimeCapture.boundaries.map((boundary) => boundary.boundaryId),
  ['layer.0.attention.q.pre_rope', 'layer.0.attention.q.post_rope']
);
assert.ok(runtimeCapture.boundaries.every((boundary) => boundary.fullTensorDigest));
assert.deepEqual(runtimeCapture.boundaries[0].samples[0].coordinate, [0, 0]);

const sourcePack = buildSourceBoundaryPack({
  identity: {
    sourceRevision: 'test',
    dtype: 'f32',
    promptDigest: 'sha256:' + '1'.repeat(64),
    modelConfigDigest: 'sha256:' + '2'.repeat(64),
    referenceScriptDigest: 'sha256:' + '3'.repeat(64),
  },
  boundaries: runtimeCapture.boundaries.map((boundary) => ({
    ...boundary,
    tolerancePolicyId: 'doppler.boundary-tolerance/source-f16-v1',
  })),
});
const deterministicTokenEvidence = {
  schema: 'doppler.deterministic-token-evidence/v1',
  exact: true,
  tokenCount: 128,
};
const passed = compareBoundaryEvidence({
  sourcePack,
  runtimeCapture,
  policy,
  deterministicTokenEvidence,
});
assert.equal(passed.promotionGate.passed, true);
assert.equal(passed.firstDivergence, null);

const divergentCapture = structuredClone(runtimeCapture);
divergentCapture.boundaries[1].samples[0].value += 1;
{
  const { digest: _digest, ...core } = divergentCapture;
  divergentCapture.digest = computeCanonicalSha256(core);
}
const failed = compareBoundaryEvidence({
  sourcePack,
  runtimeCapture: divergentCapture,
  policy,
  deterministicTokenEvidence,
});
assert.equal(failed.promotionGate.passed, false);
assert.equal(failed.firstDivergence.boundaryId, 'layer.0.attention.q.post_rope');

const quantizedWithoutControl = compareBoundaryEvidence({
  sourcePack,
  runtimeCapture,
  policy,
  artifactPrecision: 'quantized',
  deterministicTokenEvidence,
});
assert.equal(quantizedWithoutControl.promotionGate.sourcePrecisionControlPassed, false);

const tokenEvidence = buildDeterministicTokenEvidenceFromReferenceTranscript({
  schema: 'doppler.reference-transcript/v1',
  output: { tokensGenerated: 128 },
  tokens: {
    ids: Array.from({ length: 128 }, (_, index) => index),
    generatedTokenIdsHash: 'sha256:' + '4'.repeat(64),
    generatedTextHash: 'sha256:' + '5'.repeat(64),
    coverage: { mode: 'full-token-ids', omitted: 0 },
  },
});
assert.equal(tokenEvidence.schema, 'doppler.deterministic-token-evidence/v1');
assert.equal(tokenEvidence.exact, true);
assert.equal(tokenEvidence.tokenCount, 128);

const providerSourcePack = buildSourceBoundaryPackFromProviderCapture({
  schema: 'doppler.boundary-provider-capture/v1',
  provider: 'transformers',
  identity: {
    sourceRevision: 'provider-test',
    dtype: 'f32',
    promptDigest: 'sha256:' + '6'.repeat(64),
    modelConfigDigest: 'sha256:' + '7'.repeat(64),
    referenceScriptDigest: 'sha256:' + '8'.repeat(64),
  },
  runtime: { transformers: 'test' },
  boundaries: sourcePack.boundaries,
});
assert.equal(providerSourcePack.schema, 'doppler.source-boundary-pack/v1');
assert.equal(providerSourcePack.identity.provider, 'transformers');

console.log('boundary-evidence.test: ok');
