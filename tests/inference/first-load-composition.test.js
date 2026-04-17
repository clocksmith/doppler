// Regression test for `buildFirstLoadComposition` (browser-harness-suite-helpers.js).
//
// The Doppler harness emits a `firstLoad` block shaped identically to the
// transformers.js bench harness (benchmarks/runners/transformersjs-bench.js
// `buildFirstLoadComposition`). Compare-engines consumes this block to
// explain where model-load wall-clock goes. Any divergence in shape breaks
// the compare contract, so this test locks the six-field componentsMs split
// and the sum/residual semantics.

import assert from 'node:assert/strict';
import { buildFirstLoadComposition } from '../../src/inference/browser-harness-suite-helpers.js';

// Case 1: all six components present — sums and residuals resolve.
{
  const composition = buildFirstLoadComposition({
    browserLaunchMs: 70.62,
    pageReadyMs: 88.72,
    cachePrimeMs: 44727.29,
    modelLoadMs: 2864.33,
    firstTokenMs: 152.02,
    firstResponseMs: 3016.35,
  });

  assert.strictEqual(composition.schemaVersion, 1, 'schemaVersion locked at 1');
  assert.deepStrictEqual(
    Object.keys(composition.componentsMs).sort(),
    ['browserLaunchMs', 'cachePrimeMs', 'firstResponseMs', 'firstTokenMs', 'modelLoadMs', 'pageReadyMs'],
    'componentsMs must expose the six TJS-parity fields'
  );
  assert.deepStrictEqual(
    Object.keys(composition.sumsMs).sort(),
    ['endToEndFirstResponseMs', 'firstResponseFromLoadAndFirstTokenMs', 'harnessWarmStartToFirstResponseMs'],
    'sumsMs must expose the three TJS-parity sums'
  );
  assert.deepStrictEqual(
    Object.keys(composition.semantics).sort(),
    [
      'browserLaunchMs',
      'cachePrimeMs',
      'endToEndFirstResponseMs',
      'firstResponseMs',
      'firstTokenMs',
      'harnessWarmStartToFirstResponseMs',
      'modelLoadMs',
      'pageReadyMs',
    ],
    'semantics must document every component and sum field'
  );

  assert.strictEqual(
    composition.sumsMs.firstResponseFromLoadAndFirstTokenMs,
    3016.35,
    'firstResponseFromLoadAndFirstTokenMs = modelLoadMs + firstTokenMs'
  );
  assert.strictEqual(
    composition.sumsMs.harnessWarmStartToFirstResponseMs,
    47832.36,
    'harnessWarmStartToFirstResponseMs = pageReadyMs + cachePrimeMs + firstResponseMs'
  );
  assert.strictEqual(
    composition.sumsMs.endToEndFirstResponseMs,
    47902.98,
    'endToEndFirstResponseMs adds browserLaunchMs to the harness-warm-start sum'
  );
  assert.strictEqual(composition.residualsMs.firstResponseResidualMs, 0);
  assert.strictEqual(composition.consistent.firstResponse, true);
}

// Case 2: Doppler-style surface (no browserLaunch/pageReady/cachePrime yet).
// Sums that depend on missing fields must be null; modelLoad+firstToken sum
// still resolves.
{
  const composition = buildFirstLoadComposition({
    modelLoadMs: 4195.1,
    firstTokenMs: 728,
    firstResponseMs: 4923.1,
  });
  assert.strictEqual(composition.componentsMs.browserLaunchMs, null);
  assert.strictEqual(composition.componentsMs.pageReadyMs, null);
  assert.strictEqual(composition.componentsMs.cachePrimeMs, null);
  assert.strictEqual(composition.componentsMs.modelLoadMs, 4195.1);
  assert.strictEqual(composition.componentsMs.firstTokenMs, 728);
  assert.strictEqual(composition.componentsMs.firstResponseMs, 4923.1);
  assert.strictEqual(
    composition.sumsMs.firstResponseFromLoadAndFirstTokenMs,
    4923.1,
    'modelLoad-based sum resolves even when browser fields are null'
  );
  assert.strictEqual(
    composition.sumsMs.harnessWarmStartToFirstResponseMs,
    null,
    'warm-start sum is null while cachePrime/pageReady are not measured'
  );
  assert.strictEqual(
    composition.sumsMs.endToEndFirstResponseMs,
    null,
    'end-to-end sum is null while browserLaunchMs is not measured'
  );
  assert.strictEqual(composition.residualsMs.firstResponseResidualMs, 0);
  assert.strictEqual(composition.consistent.firstResponse, true);
}

// Case 3: residual above the 2ms tolerance flips consistent.firstResponse.
{
  const composition = buildFirstLoadComposition({
    modelLoadMs: 1000,
    firstTokenMs: 100,
    firstResponseMs: 1105,
  });
  assert.strictEqual(composition.residualsMs.firstResponseResidualMs, 5);
  assert.strictEqual(composition.consistent.firstResponse, false);
}

// Case 4: missing modelLoadMs/firstTokenMs nulls both the sum and the
// consistency check (no phantom zeros).
{
  const composition = buildFirstLoadComposition({
    firstResponseMs: 500,
  });
  assert.strictEqual(composition.componentsMs.modelLoadMs, null);
  assert.strictEqual(composition.componentsMs.firstTokenMs, null);
  assert.strictEqual(composition.sumsMs.firstResponseFromLoadAndFirstTokenMs, null);
  assert.strictEqual(composition.residualsMs.firstResponseResidualMs, null);
  assert.strictEqual(composition.consistent.firstResponse, null);
}

console.log('first-load-composition.test: ok');
