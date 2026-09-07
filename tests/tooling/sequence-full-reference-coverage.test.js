import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { assertCompleteSequenceReferences } from '../../tools/qualify-installed-sequence.js';

const partial = JSON.parse(await fs.readFile(new URL('../../tools/data/esm2-t12-35m-ur50d-sequence-reference.json', import.meta.url)));
assert.throws(() => assertCompleteSequenceReferences([partial], 480), /Every pooled/,
  'Historical partial probes must not silently establish full-output coverage.');
const full = structuredClone(partial);
const indices = Array.from({ length: 480 }, (_, index) => index);
full.probes.pooledEmbedding = { indices, values: Array(480).fill(0) };
full.probes.tokenEmbeddings = full.input.tokenIds.map((_, position) => ({ position, indices, values: Array(480).fill(0) }));
assertCompleteSequenceReferences([full], 480);
for (const mutate of [
  ref => ref.probes.tokenEmbeddings.pop(),
  ref => { ref.probes.tokenEmbeddings[1].position = 0; },
  ref => { ref.probes.tokenEmbeddings[0].indices = [0]; },
  ref => { ref.probes.pooledEmbedding.values[3] = NaN; },
  ref => { ref.probes.tokenEmbeddings[0].values.pop(); },
]) {
  const changed = structuredClone(full); mutate(changed);
  assert.throws(() => assertCompleteSequenceReferences([changed], 480));
}
console.log('sequence-full-reference-coverage.test: ok (coverage contract, not inference evidence)');
