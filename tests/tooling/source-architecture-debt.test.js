import assert from 'node:assert/strict';

import {
  findArchitecturePolicyRelaxations,
  validateArchitecturePolicyDelta,
  validateSoftLimitReviewDelta,
} from '../../tools/lib/source-architecture-debt.js';
import { resolvePolicyBaseRef } from '../../tools/lib/policy-base.js';

const digest = `sha256:${'a'.repeat(64)}`;
const baseline = {
  softLimitReviews: {
    'inference/existing.js': { reviewedLines: 800 },
  },
};

assert.deepEqual(validateSoftLimitReviewDelta(baseline, baseline), []);
assert.match(
  validateSoftLimitReviewDelta({
    softLimitReviews: {
      ...baseline.softLimitReviews,
      'inference/new.js': { reviewedLines: 760 },
    },
  }, baseline)[0],
  /adds a soft-limit review/
);
assert.match(
  validateSoftLimitReviewDelta({
    softLimitReviews: {
      'inference/existing.js': { reviewedLines: 801 },
    },
  }, baseline)[0],
  /raises reviewedLines from 800 to 801/
);
assert.deepEqual(validateSoftLimitReviewDelta({
  softLimitReviews: {
    'inference/existing.js': { reviewedLines: 801, debtAuthorization: digest },
  },
}, baseline, digest), []);

const baselinePolicy = {
  softLineLimit: 750,
  lineLimit: 999,
  allowedDependencies: { inference: ['config', 'inference'] },
  dependencyExceptions: [],
  experimentalBridges: {},
  reachability: { standaloneModules: {} },
  facades: [],
  compatibilityOnlyOwners: ['utils'],
  constitutionalDomains: { runtime: ['client/runtime.js'] },
  constitutionalImportGraphs: [{
    domain: 'runtime',
    entryPoints: ['client/runtime.js'],
    forbiddenPathPrefixes: ['converter/'],
  }],
  softLimitReviews: {},
  legacyOversize: {},
};
const relaxedPolicy = {
  ...baselinePolicy,
  lineLimit: 1000,
  allowedDependencies: { inference: ['config', 'converter', 'inference'] },
  facades: ['inference/escape.js'],
};
assert.deepEqual(findArchitecturePolicyRelaxations(relaxedPolicy, baselinePolicy), [
  'lineLimit increased from 999 to 1000',
  'allowed dependency added inference->converter',
  'facade exception added inference/escape.js',
]);
assert.equal(validateArchitecturePolicyDelta(relaxedPolicy, baselinePolicy).length, 3);
assert.deepEqual(validateArchitecturePolicyDelta({
  ...relaxedPolicy,
  debtAuthorization: digest,
}, baselinePolicy, digest), []);
assert.equal(validateArchitecturePolicyDelta({
  ...relaxedPolicy,
  debtAuthorization: digest,
}, { ...baselinePolicy, debtAuthorization: digest }, digest).length, 3);

const renamedPolicy = {
  ...baselinePolicy,
  constitutionalRenames: { domains: { runtime: 'execution' }, files: { 'client/runtime.js': 'client/execution.js' } },
  constitutionalDomains: { execution: ['client/execution.js'] },
  constitutionalImportGraphs: [{ domain: 'execution', entryPoints: ['client/execution.js'], forbiddenPathPrefixes: ['converter/'] }],
};
assert.deepEqual(validateArchitecturePolicyDelta(renamedPolicy, baselinePolicy), []);
assert.deepEqual(validateArchitecturePolicyDelta(renamedPolicy, renamedPolicy), []);
const droppedBoundary = structuredClone(renamedPolicy);
droppedBoundary.constitutionalImportGraphs[0].forbiddenPathPrefixes = [];
assert.match(validateArchitecturePolicyDelta(droppedBoundary, baselinePolicy).join('; '), /forbidden prefix removed converter/);
const droppedEntry = structuredClone(renamedPolicy);
droppedEntry.constitutionalImportGraphs[0].entryPoints = [];
assert.match(validateArchitecturePolicyDelta(droppedEntry, baselinePolicy).join('; '), /entry point removed client\/execution.js/);
const movedOwner = structuredClone(renamedPolicy);
movedOwner.constitutionalRenames.files['client/runtime.js'] = 'experimental/runtime.js';
assert.match(validateArchitecturePolicyDelta(movedOwner, baselinePolicy).join('; '), /same owner directory/);
const mergedFiles = structuredClone(renamedPolicy);
mergedFiles.constitutionalRenames.files['client/another.js'] = 'client/execution.js';
assert.match(validateArchitecturePolicyDelta(mergedFiles, baselinePolicy).join('; '), /one-to-one/);

assert.equal(resolvePolicyBaseRef([], {}), 'HEAD');
assert.equal(resolvePolicyBaseRef([], { DOPPLER_POLICY_BASE_REF: '' }), 'HEAD');
assert.equal(resolvePolicyBaseRef([], { DOPPLER_POLICY_BASE_REF: '000000' }), 'HEAD');
assert.equal(resolvePolicyBaseRef([], { DOPPLER_POLICY_BASE_REF: 'base-sha' }), 'base-sha');
assert.equal(resolvePolicyBaseRef(['--base', 'explicit'], {}), 'explicit');

console.log('source-architecture-debt.test: ok');
