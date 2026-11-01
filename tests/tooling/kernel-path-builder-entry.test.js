import assert from 'node:assert/strict';
import { buildKernelPathBuilderArtifacts, buildKernelPathBuilderProposals } from '../../src/tooling/kernel-path-builder/index.js';

const artifacts = buildKernelPathBuilderArtifacts({ configEntries: [], manifestEntries: [], registryEntries: [] });
assert.equal(artifacts.index.source, 'kernel-path-builder');
assert.equal(artifacts.index.stats.models, 0);
assert.deepEqual(artifacts.index.models, []);
assert.deepEqual(artifacts.proposals, buildKernelPathBuilderProposals(artifacts.index));
assert.equal(artifacts.proposals.stats.proposals, 0);
assert.equal(typeof artifacts.reportMarkdown, 'string');
assert.ok(artifacts.reportMarkdown.length > 0);
console.log('kernel-path-builder-entry.test: ok');
