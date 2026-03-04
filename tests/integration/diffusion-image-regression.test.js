import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  assertImageRegressionWithinTolerance,
  computeImageFingerprint,
  computeImageRegressionMetrics,
} from '../../src/diffusion/image-regression.js';

const fixturesPath = resolve(
  process.cwd(),
  'tests/fixtures/diffusion/sd3-golden-fixtures.json'
);
const fixtureDoc = JSON.parse(readFileSync(fixturesPath, 'utf8'));

assert.equal(fixtureDoc.schemaVersion, 1);
assert.ok(Array.isArray(fixtureDoc.fixtures));
assert.ok(fixtureDoc.fixtures.length > 0);

for (const fixture of fixtureDoc.fixtures) {
  const pixels = Uint8Array.from(fixture.pixels || []);
  assert.equal(
    computeImageFingerprint(pixels),
    fixture.fingerprint,
    `fixture ${fixture.id}: fingerprint mismatch`
  );

  const exactMetrics = computeImageRegressionMetrics(pixels, pixels);
  assert.equal(exactMetrics.samples, pixels.length);
  assert.equal(exactMetrics.maxAbsDiff, 0);
  assert.equal(exactMetrics.mae, 0);
  assert.equal(exactMetrics.rmse, 0);
  assert.equal(exactMetrics.psnr, Infinity);

  const fixtureMetrics = assertImageRegressionWithinTolerance(
    pixels,
    pixels,
    fixture.tolerance,
    `fixture ${fixture.id}`
  );
  assert.equal(fixtureMetrics.maxAbsDiff, 0);

  const shifted = Uint8Array.from(pixels);
  shifted[0] = Math.max(0, shifted[0] - 1);
  assert.throws(
    () => assertImageRegressionWithinTolerance(
      shifted,
      pixels,
      fixture.tolerance,
      `fixture ${fixture.id} strict`
    ),
    /exceeded limit|below minimum/
  );

  const relaxed = assertImageRegressionWithinTolerance(
    shifted,
    pixels,
    {
      maxAbsDiff: 1,
      mae: 1,
      rmse: 1,
      minPsnr: 30,
    },
    `fixture ${fixture.id} relaxed`
  );
  assert.equal(relaxed.maxAbsDiff, 1);
}

console.log('diffusion-image-regression.test: ok');
