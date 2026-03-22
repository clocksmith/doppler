import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import { buildKernelPathBuilderArtifactsPayload } from '../../tools/sync-kernel-path-builder-index.js';

const artifacts = await buildKernelPathBuilderArtifactsPayload();
const expectedIndex = `${JSON.stringify(artifacts.index, null, 2)}\n`;
const expectedProposals = `${JSON.stringify(artifacts.proposals, null, 2)}\n`;
const expectedReport = artifacts.reportMarkdown.endsWith('\n')
  ? artifacts.reportMarkdown
  : `${artifacts.reportMarkdown}\n`;

const currentIndex = await fs.readFile(new URL('../../demo/data/kernel-path-builder-index.json', import.meta.url), 'utf8');
const currentProposals = await fs.readFile(new URL('../../demo/data/kernel-path-builder-proposals.json', import.meta.url), 'utf8');
const currentReport = await fs.readFile(new URL('../../demo/data/kernel-path-builder-report.md', import.meta.url), 'utf8');

assert.equal(currentIndex, expectedIndex);
assert.equal(currentProposals, expectedProposals);
assert.equal(currentReport, expectedReport);

console.log('kernel-path-builder-index-contract.test: ok');
