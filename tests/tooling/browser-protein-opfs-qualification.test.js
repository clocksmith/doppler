import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  assertProteinSequenceSmokeResult,
  parseArgs,
} from '../../tools/ci-browser-opfs-registry-smoke.js';

const fixturePath = path.resolve('tools/data/esm2-35m.browser-qualification.json');
const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
const reference = JSON.parse(await readFile(
  path.resolve('tools/data/esm2-t12-35m-ur50d-sequence-reference.json'),
  'utf8'
));
fixture.reference = reference;

const pooledEmbedding = Array.from({ length: fixture.expectedEmbeddingDim }, () => 0);
for (let index = 0; index < reference.probes.pooledEmbedding.indices.length; index += 1) {
  pooledEmbedding[reference.probes.pooledEmbedding.indices[index]] = reference.probes.pooledEmbedding.values[index];
}
const tokenEmbeddingProbes = reference.probes.tokenEmbeddings.map((probe) => {
  const values = Array.from({ length: fixture.expectedEmbeddingDim }, () => 0);
  for (let index = 0; index < probe.indices.length; index += 1) {
    values[probe.indices[index]] = probe.values[index];
  }
  return { position: probe.position, values };
});
const response = {
  ok: true,
  result: {
    output: {
      mode: 'sequence',
      model: {
        modelId: reference.modelId,
        sourceCheckpointId: reference.source.checkpointId,
      },
      input: {
        sequence: reference.input.sequence,
        alphabet: reference.input.alphabet,
      },
      tokens: reference.input.tokenIds,
      embeddingDim: fixture.expectedEmbeddingDim,
      pooledEmbedding,
      tokenEmbeddingProbes,
      finite: {
        pooledEmbedding: true,
        tokenEmbeddings: true,
      },
    },
    metrics: {
      modelLoadMs: 1,
      sequenceEncodingMs: 1,
    },
  },
};

assert.doesNotThrow(() => assertProteinSequenceSmokeResult('fixture', response, fixture));

const mismatchedTokenResponse = structuredClone(response);
mismatchedTokenResponse.result.output.tokens[1] = 0;
assert.throws(
  () => assertProteinSequenceSmokeResult('fixture', mismatchedTokenResponse, fixture),
  /tokenizer ids: mismatch/
);

const args = parseArgs(['--model-id', 'esm2-t12-35m-ur50d-f32-af32']);
assert.match(args.profileDir, /ci-opfs\/esm2-t12-35m-ur50d-f32-af32$/u);

console.log('browser-protein-opfs-qualification.test: ok');
