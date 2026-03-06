import assert from 'node:assert/strict';
import { buildContractSummary } from '../../tools/check-contract-artifacts.js';

const summary = await buildContractSummary({
  json: true,
  reportsRoot: '',
  failOnReportContracts: false,
  withLean: false,
  leanCheck: true,
  leanManifestRoot: 'models',
  leanConfigRoot: 'tools/configs/conversion',
  leanFixtureMap: 'tools/configs/conversion/lean-execution-contract-fixtures.json',
  leanRequireManifestMatch: false,
});
assert.equal(summary.schemaVersion, 1);
assert.equal(summary.source, 'doppler');
assert.equal(summary.ok, true);
assert.equal(Array.isArray(summary.artifacts), true);
assert.equal(summary.artifacts.some((entry) => entry.id === 'kernelPath' && entry.ok === true), true);
assert.equal(summary.artifacts.some((entry) => entry.id === 'layerPattern' && entry.ok === true), true);

const leanSummary = await buildContractSummary({
  json: true,
  reportsRoot: '',
  failOnReportContracts: false,
  withLean: true,
  leanCheck: false,
  leanManifestRoot: 'models',
  leanConfigRoot: 'tools/configs/conversion',
  leanFixtureMap: 'tools/configs/conversion/lean-execution-contract-fixtures.json',
  leanRequireManifestMatch: true,
});
assert.equal(leanSummary.ok, true);
assert.equal(leanSummary.lean?.manifestSweep?.ok, true);
assert.equal(leanSummary.lean?.configSweep?.ok, true);
assert.equal(
  leanSummary.artifacts.some((entry) => entry.id === 'leanExecutionContractManifests' && entry.ok === true),
  true
);
assert.equal(
  leanSummary.artifacts.some((entry) => entry.id === 'leanExecutionContractConfigs' && entry.ok === true),
  true
);

console.log('check-contract-artifacts.test: ok');
