import assert from 'node:assert/strict';

import { runEmbeddingSemanticChecks } from '../../src/inference/browser-harness-text-helpers.js';

const vectors = new Map([
  ['task: search result | query: Find a quiet place to borrow books.', Float32Array.from([1, 0])],
  ['title: None | text: The public library lends books and has silent study rooms.', Float32Array.from([1, 0])],
  ['title: None | text: The bakery sells bread and cakes.', Float32Array.from([0, 1])],
  ['title: None | text: The auto shop repairs brakes and tires.', Float32Array.from([-1, 0])],
  ['task: search result | query: Cancel my subscription before it renews.', Float32Array.from([0, 1])],
  ['task: search result | query: Stop the plan so it does not renew.', Float32Array.from([0, 1])],
  ['task: search result | query: Rain is expected along the coast tonight.', Float32Array.from([1, 0])],
]);

const pipeline = {
  manifest: {
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
  },
  runtimeConfig: {},
  reset() {},
  async embed(text) {
    const embedding = vectors.get(text);
    assert.ok(embedding, `Unexpected semantic input: ${text}`);
    return { embedding };
  },
};

const semantic = await runEmbeddingSemanticChecks(pipeline, {
  embeddingSemantic: {
    retrievalCases: [
      {
        id: 'library_lookup',
        query: 'Find a quiet place to borrow books.',
        docs: [
          'The public library lends books and has silent study rooms.',
          'The bakery sells bread and cakes.',
          'The auto shop repairs brakes and tires.',
        ],
        expectedDoc: 0,
      },
    ],
    pairCases: [
      {
        id: 'cancel_plan',
        anchor: 'Cancel my subscription before it renews.',
        positive: 'Stop the plan so it does not renew.',
        negative: 'Rain is expected along the coast tonight.',
      },
    ],
    minRetrievalTop1Acc: 1,
    minPairAcc: 1,
    pairMargin: 0.1,
  },
});

assert.equal(semantic.passed, true);
assert.equal(semantic.style, 'embeddinggemma');

assert.deepEqual(semantic.retrieval[0], {
  id: 'library_lookup',
  query: 'Find a quiet place to borrow books.',
  formattedQuery: 'task: search result | query: Find a quiet place to borrow books.',
  docs: [
    {
      text: 'The public library lends books and has silent study rooms.',
      formattedText: 'title: None | text: The public library lends books and has silent study rooms.',
    },
    {
      text: 'The bakery sells bread and cakes.',
      formattedText: 'title: None | text: The bakery sells bread and cakes.',
    },
    {
      text: 'The auto shop repairs brakes and tires.',
      formattedText: 'title: None | text: The auto shop repairs brakes and tires.',
    },
  ],
  passed: true,
  expectedDoc: 0,
  topDoc: 0,
  sims: [1, 0, -1],
});

assert.deepEqual(semantic.pairs[0], {
  id: 'cancel_plan',
  anchor: 'Cancel my subscription before it renews.',
  formattedAnchor: 'task: search result | query: Cancel my subscription before it renews.',
  positive: 'Stop the plan so it does not renew.',
  formattedPositive: 'task: search result | query: Stop the plan so it does not renew.',
  negative: 'Rain is expected along the coast tonight.',
  formattedNegative: 'task: search result | query: Rain is expected along the coast tonight.',
  passed: true,
  simPos: 1,
  simNeg: 0,
  margin: 1,
});

console.log('embedding-semantic-details.test: ok');
