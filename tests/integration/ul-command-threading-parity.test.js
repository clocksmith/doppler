import assert from 'node:assert/strict';

import { normalizeNodeCommand } from '../../src/tooling/node-command-runner.js';
import { normalizeBrowserCommand } from '../../src/tooling/browser-command-runner.js';

const raw = {
  command: 'bench',
  workloadType: 'training',
  modelId: null,
  trainingTests: ['ul-stage2'],
  trainingStage: 'stage2_base',
  trainingConfig: {
    ul: {
      enabled: true,
      stage: 'stage2_base',
    },
  },
  stage1Artifact: '/tmp/ul_stage1_manifest.json',
  stage1ArtifactHash: 'abc123',
  ulArtifactDir: '/tmp/ul',
  trainingSchemaVersion: 1,
  trainingBenchSteps: 7,
};

const nodeRequest = normalizeNodeCommand(raw);
const browserRequest = normalizeBrowserCommand(raw);

assert.deepEqual(browserRequest, nodeRequest);
assert.equal(nodeRequest.command, 'bench');
assert.equal(nodeRequest.suite, 'bench');
assert.equal(nodeRequest.workloadType, 'training');
assert.equal(nodeRequest.trainingSchemaVersion, 1);
assert.equal(nodeRequest.trainingBenchSteps, 7);
assert.equal(nodeRequest.trainingStage, 'stage2_base');
assert.equal(nodeRequest.stage1Artifact, '/tmp/ul_stage1_manifest.json');
assert.equal(nodeRequest.stage1ArtifactHash, 'abc123');
assert.equal(nodeRequest.ulArtifactDir, '/tmp/ul');

console.log('ul-command-threading-parity.test: ok');
