import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

import { getDefaultEmbeddingSemanticFixtures } from '../../src/inference/browser-harness-text-helpers.js';

const asset = JSON.parse(
  readFileSync(new URL('../../src/inference/fixtures/embedding-semantic-fixtures.json', import.meta.url), 'utf8')
);
const defaults = getDefaultEmbeddingSemanticFixtures();

assert.ok(Number.isInteger(asset.schemaVersion));
assert.ok(asset.schemaVersion >= 1);
assert.deepEqual(defaults, asset.defaults);
assert.equal(defaults.minRetrievalTop1Acc, asset.defaults.minRetrievalTop1Acc);
assert.equal(defaults.minPairAcc, asset.defaults.minPairAcc);
assert.equal(defaults.pairMargin, asset.defaults.pairMargin);
assert.equal(defaults.retrievalCases.length, asset.defaults.retrievalCases.length);
assert.equal(defaults.pairCases.length, asset.defaults.pairCases.length);

assert.deepEqual(defaults.retrievalCases[0], {
  id: 'library_search',
  query: 'Where can I borrow books and study quietly?',
  docs: [
    'The city library lends books, provides study rooms, and offers free Wi-Fi.',
    'The cafe serves coffee, pastries, and sandwiches all day.',
    'The bike repair shop fixes flat tires and broken chains.',
  ],
  expectedDoc: 0,
});

assert.deepEqual(defaults.pairCases[0], {
  id: 'bike_paraphrase',
  anchor: 'The child is riding a bicycle through the park.',
  positive: 'A kid bikes along a path in the park.',
  negative: 'The stock market closed lower after interest-rate news.',
});

console.log('embedding-semantic-fixtures-contract.test: ok');
