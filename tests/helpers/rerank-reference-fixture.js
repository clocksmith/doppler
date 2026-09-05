import { computeCanonicalSha256 } from '../../src/formats/canonical-hash.js';

// Synthetic contract evidence, not a source model or physical execution claim.
export function createRerankReferenceFixture() {
  const digest = `sha256:${'1'.repeat(64)}`;
  const reference = {
    schema: 'doppler.rerank-source-reference/v1',
    source: { checkpointId: 'fixture/model', repository: 'fixture/model', revision: '1'.repeat(40),
      engine: 'synthetic-test', files: [{ path: 'weights.fixture', hash: digest }] },
    input: { query: 'query', documents: ['one', 'two'] }, scoringConfig: { score: 'true_logit' },
    tolerances: { logitMaxAbs: 0.1, scoreMaxAbs: 0.1, probabilityMaxAbs: 0.01, ranking: 'exact' },
    outputs: [
      { index: 0, document: 'one', tokenIds: [1, 2], trueLogit: 1, falseLogit: 0, score: 1, probability: 0.7310585786300049 },
      { index: 1, document: 'two', tokenIds: [1, 3], trueLogit: 0, falseLogit: 1, score: 0, probability: 0.5 },
    ],
  };
  return {
    schema: 'doppler.rerank-reference-transcript/v1', operation: 'rerank', modelId: 'fixture-model',
    manifestHash: digest, executionGraphHash: digest, surface: 'test-webgpu',
    source: { kind: 'synthetic-test', path: 'report.json', hash: digest },
    reference, referenceDigest: computeCanonicalSha256(reference),
    observation: structuredClone({ input: reference.input, scoringConfig: reference.scoringConfig, outputs: reference.outputs }),
  };
}
