import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

// Synthetic reference values exercise admission, not physical model accuracy.
export function createEmbeddingReferenceFixture() {
  const digest = `sha256:${'1'.repeat(64)}`;
  const reference = {
    schema: 'doppler.embedding-source-reference/v1',
    source: { checkpointId: 'fixture/model', repository: 'fixture/model', revision: '1'.repeat(40),
      engine: 'synthetic-test', files: [{ path: 'weights.fixture', hash: digest }] },
    input: { texts: ['one', 'two'] },
    embeddingContract: { dimension: 4, postprocessor: {
      poolingMode: 'last', includePrompt: true, projections: [], normalize: 'l2',
    } },
    tolerances: { embeddingMaxAbs: 0.01, tokenIds: 'exact' },
    outputs: [
      { text: 'one', tokenIds: [1, 2], embedding: [1, 0, 0, 0] },
      { text: 'two', tokenIds: [1, 3], embedding: [0, 1, 0, 0] },
    ],
  };
  return {
    schema: 'doppler.embedding-reference-transcript/v1', operation: 'embed', modelId: 'fixture-model',
    manifestHash: digest, executionGraphHash: digest, surface: 'test-webgpu',
    source: { kind: 'synthetic-test', path: 'report.json', hash: digest },
    reference, referenceDigest: computeCanonicalSha256(reference),
    observation: structuredClone({ input: reference.input,
      embeddingContract: reference.embeddingContract, outputs: reference.outputs }),
  };
}
