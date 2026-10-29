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

const facadeBaseline = {
  ...baselinePolicy,
  facades: ['outer.js', 'inner.js'],
  facadeDependencies: { 'outer.js': ['inner.js'], 'inner.js': ['a/x.js', 'a/y.js'] },
};
const inlinedFacade = {
  ...facadeBaseline,
  facades: ['outer.js'],
  facadeDependencies: { 'outer.js': ['a/x.js', 'a/y.js'] },
};
assert.deepEqual(findArchitecturePolicyRelaxations(inlinedFacade, facadeBaseline), []);
const widenedFacade = structuredClone(inlinedFacade);
widenedFacade.facadeDependencies['outer.js'].push('a/z.js');
assert.deepEqual(findArchitecturePolicyRelaxations(widenedFacade, facadeBaseline), ['facade dependency added outer.js->a/z.js']);

const forwarderBaseline = {
  ...baselinePolicy,
  facades: ['outer.js'],
  facadeDependencies: { 'outer.js': ['config/index.js'] },
};
const inlinedForwarder = {
  ...forwarderBaseline,
  facadeDependencies: { 'outer.js': ['config/schema/index.js'] },
};
assert.deepEqual(findArchitecturePolicyRelaxations(inlinedForwarder, forwarderBaseline),
  ['facade dependency added outer.js->config/schema/index.js']);
assert.deepEqual(findArchitecturePolicyRelaxations(inlinedForwarder, forwarderBaseline,
  { 'config/index.js': ['config/schema/index.js'] }), []);
assert.equal(validateArchitecturePolicyDelta(inlinedForwarder, forwarderBaseline, null,
  { 'config/index.js': ['config/runtime.js'] }).length, 1);

const compatBaseline = {
  ...baselinePolicy,
  topLevelOwners: { inference: 'x', loaders: 'x', utils: 'x' },
  compatibilityOnlyOwners: ['loaders', 'utils'],
};
const liftedCompat = { ...compatBaseline, compatibilityOnlyOwners: ['utils'] };
assert.deepEqual(findArchitecturePolicyRelaxations(liftedCompat, compatBaseline),
  ['compatibility-only restriction removed loaders']);
const deletedCompatOwner = { ...liftedCompat, topLevelOwners: { inference: 'x', utils: 'x' } };
assert.deepEqual(findArchitecturePolicyRelaxations(deletedCompatOwner, compatBaseline), []);

// Cross-directory moves re-register existing entries at their new paths.
const moveBaseline = {
  ...baselinePolicy,
  facades: ['inference/harness.js', 'tooling-exports.js'],
  facadeDependencies: {
    'inference/harness.js': ['inference/harness/a.js'],
    'tooling-exports.js': ['inference/harness.js', 'tooling/command.js'],
  },
  dependencyExceptions: [{ from: 'inference/harness/a.js', toOwner: 'client' }],
  softLimitReviews: { 'inference/harness/a.js': { reviewedLines: 870 } },
  genericModuleReviews: { 'inference/harness.js': { classification: 'compatibility-facade' } },
};
const moves = {
  'inference/harness.js': 'tooling/harness.js',
  'inference/harness/a.js': 'tooling/harness/a.js',
};
const movedPolicy = {
  ...moveBaseline,
  movedPaths: moves,
  facades: ['tooling/harness.js', 'tooling-exports.js'],
  facadeDependencies: {
    'tooling/harness.js': ['tooling/harness/a.js'],
    'tooling-exports.js': ['tooling/command.js', 'tooling/harness.js'],
  },
  dependencyExceptions: [{ from: 'tooling/harness/a.js', toOwner: 'client' }],
  softLimitReviews: { 'tooling/harness/a.js': { reviewedLines: 870 } },
  genericModuleReviews: { 'tooling/harness.js': { classification: 'compatibility-facade' } },
};
const unmappedMove = structuredClone(movedPolicy);
delete unmappedMove.movedPaths;
assert.deepEqual(findArchitecturePolicyRelaxations(unmappedMove, moveBaseline), [
  'dependency exception added tooling/harness/a.js->client',
  'facade exception added tooling/harness.js',
  'facade dependency added tooling-exports.js->tooling/harness.js',
]);
assert.match(validateArchitecturePolicyDelta(unmappedMove, moveBaseline).join('; '), /adds a soft-limit review/);
assert.deepEqual(validateArchitecturePolicyDelta(movedPolicy, moveBaseline), []);
const moveFiles = new Map([
  ['inference/harness.js', { current: false, baseline: true }],
  ['inference/harness/a.js', { current: false, baseline: true }],
  ['tooling/harness.js', { current: true, baseline: false }],
  ['tooling/harness/a.js', { current: true, baseline: false }],
]);
const fileState = (relativePath) => moveFiles.get(relativePath) ?? { current: false, baseline: false };
assert.deepEqual(validateArchitecturePolicyDelta(movedPolicy, moveBaseline, null, {}, fileState), []);
// Once the move is in the baseline the entries are inert, not new debt.
const inertState = (relativePath) => ({ current: relativePath.startsWith('tooling/'), baseline: relativePath.startsWith('tooling/') });
assert.deepEqual(validateArchitecturePolicyDelta(movedPolicy, { ...movedPolicy, movedPaths: undefined }, null, {}, inertState), []);
const stillPresent = (relativePath) => ({ ...fileState(relativePath), current: true });
assert.match(validateArchitecturePolicyDelta(movedPolicy, moveBaseline, null, {}, stillPresent).join('; '),
  /inference\/harness\.js still exists/);
const intoExisting = (relativePath) => ({ ...fileState(relativePath), baseline: true });
assert.match(validateArchitecturePolicyDelta(movedPolicy, moveBaseline, null, {}, intoExisting).join('; '),
  /already existed at the base revision/);
const widenedReview = structuredClone(movedPolicy);
widenedReview.softLimitReviews['tooling/harness/a.js'].reviewedLines = 900;
assert.match(validateArchitecturePolicyDelta(widenedReview, moveBaseline).join('; '),
  /tooling\/harness\/a\.js: raises reviewedLines from 870 to 900/);
const widenedEdges = structuredClone(movedPolicy);
widenedEdges.facadeDependencies['tooling/harness.js'].push('inference/rerank.js');
assert.match(validateArchitecturePolicyDelta(widenedEdges, moveBaseline).join('; '),
  /facade dependency added tooling\/harness\.js->inference\/rerank\.js/);
const widenedException = structuredClone(movedPolicy);
widenedException.dependencyExceptions.push({ from: 'tooling/harness/a.js', toOwner: 'cli' });
assert.match(validateArchitecturePolicyDelta(widenedException, moveBaseline).join('; '),
  /dependency exception added tooling\/harness\/a\.js->cli/);
const reclassified = structuredClone(movedPolicy);
reclassified.genericModuleReviews['tooling/harness.js'].classification = 'semantic-owner';
assert.match(validateArchitecturePolicyDelta(reclassified, moveBaseline).join('; '),
  /changes classification from compatibility-facade to semantic-owner/);
const mergedMove = { ...movedPolicy, movedPaths: { ...moves, 'inference/other.js': 'tooling/harness.js' } };
assert.match(validateArchitecturePolicyDelta(mergedMove, moveBaseline).join('; '), /one-to-one/);
const chainedMove = { ...movedPolicy, movedPaths: { ...moves, 'tooling/harness.js': 'tooling/other.js' } };
assert.match(validateArchitecturePolicyDelta(chainedMove, moveBaseline).join('; '), /without chains/);
const ontoExistingEntry = {
  ...movedPolicy,
  movedPaths: { 'inference/harness/a.js': 'tooling-exports.js' },
};
assert.match(validateArchitecturePolicyDelta(ontoExistingEntry, moveBaseline).join('; '),
  /merges into an existing policy entry/);

assert.equal(resolvePolicyBaseRef([], {}), 'HEAD');
assert.equal(resolvePolicyBaseRef([], { DOPPLER_POLICY_BASE_REF: '' }), 'HEAD');
assert.equal(resolvePolicyBaseRef([], { DOPPLER_POLICY_BASE_REF: '000000' }), 'HEAD');
assert.equal(resolvePolicyBaseRef([], { DOPPLER_POLICY_BASE_REF: 'base-sha' }), 'base-sha');
assert.equal(resolvePolicyBaseRef(['--base', 'explicit'], {}), 'explicit');

console.log('source-architecture-debt.test: ok');
