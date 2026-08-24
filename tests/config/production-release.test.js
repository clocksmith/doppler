import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  assertProductionRelease,
  hashProductionRelease,
  validateProductionRelease,
} from '../../src/config/production-release.js';

const fixture = JSON.parse(await fs.readFile(
  'tests/fixtures/production-release/electron-document-search-reranker.json',
  'utf8'
));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

{
  const validation = validateProductionRelease(fixture);
  assert.equal(validation.ok, true, validation.errors.join('\n'));
  assert.equal(assertProductionRelease(fixture), fixture);
  assert.match(hashProductionRelease(fixture), /^sha256:[0-9a-f]{64}$/);
  assert.equal(fixture.evidenceClass, 'reference-fixture');
  assert.equal(fixture.claimBoundary.externalCustomer, false);
  assert.equal(fixture.claimBoundary.commercialClaimAllowed, false);
  assert.equal(fixture.rollout.activationAuthority, 'customer');
  assert.equal(fixture.rollout.selfPromotionAllowed, false);
}

{
  const broken = clone(fixture);
  broken.application.platform = 'browser';
  const validation = validateProductionRelease(broken);
  assert.ok(validation.errors.includes('application.platform must be "electron".'));
}

{
  const broken = clone(fixture);
  broken.supportedDevices.targets = broken.supportedDevices.targets.filter((target) => target.os !== 'macos');
  const validation = validateProductionRelease(broken);
  assert.ok(validation.errors.includes('supportedDevices.targets must include Windows and macOS targets.'));
}

{
  const broken = clone(fixture);
  broken.rollout.selfPromotionAllowed = true;
  const validation = validateProductionRelease(broken);
  assert.ok(validation.errors.includes('rollout.selfPromotionAllowed must be false.'));
}

{
  const broken = clone(fixture);
  broken.claimBoundary.externalCustomer = true;
  const validation = validateProductionRelease(broken);
  assert.ok(validation.errors.includes(
    'reference-fixture releases cannot claim an external customer or commercial evidence.'
  ));
}

console.log('production-release.test: ok');
