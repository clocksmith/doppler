import assert from 'node:assert/strict';

import {
  buildTokenTraceRecord,
  probToPerplexity,
  summarizePerplexityRecords,
} from '../../demo/ui/token-press/metrics.js';

{
  assert.equal(probToPerplexity(1), 1);
  assert.equal(probToPerplexity(0.5), 2);
}

{
  const record = buildTokenTraceRecord({
    tokenId: 7,
    text: 'Yes',
    confidence: 0.5,
    topK: [
      { token: 7, text: 'Yes', logit: 11.25, prob: 0.5 },
      { token: 11, text: 'No', logit: 10.75, prob: 0.3 },
      { token: 29, text: 'Maybe', logit: 9.9, prob: 0.2 },
    ],
  }, 3);

  assert.deepEqual(
    {
      index: record.index,
      tokenId: record.tokenId,
      text: record.text,
      confidence: record.confidence,
      perplexity: record.perplexity,
      selectedRank: record.selectedRank,
      selectedLogit: record.selectedLogit,
    },
    {
      index: 3,
      tokenId: 7,
      text: 'Yes',
      confidence: 0.5,
      perplexity: 2,
      selectedRank: 1,
      selectedLogit: 11.25,
    }
  );
  assert.equal(record.topK[1].probability, 0.3);
  assert.equal(record.topK[1].perplexity, probToPerplexity(0.3));
}

{
  const summary = summarizePerplexityRecords([
    { perplexity: 1.1 },
    { perplexity: 1.2 },
    { perplexity: 1.3 },
    { perplexity: 1.4 },
    { perplexity: 1.5 },
    { perplexity: 1.6 },
    { perplexity: 1.7 },
    { perplexity: 20 },
    { perplexity: 40 },
    { perplexity: 1000 },
  ]);

  assert.equal(summary.count, 10);
  assert.equal(summary.min, 1.1);
  assert.equal(summary.max, 1000);
  assert.ok(summary.displayMax < summary.max, 'display range should clip extreme perplexity outliers');
  assert.ok(summary.extremeHighCount >= 1, 'summary should flag out-of-range high perplexity tokens');
}

console.log('token-press-metrics.test: ok');
