import assert from 'node:assert/strict';

import { normalizeNodeCommand } from '../../src/tooling/node-command-runner.js';
import { normalizeBrowserCommand } from '../../src/tooling/browser-command-runner.js';

const raw = {
  command: 'bench',
  workloadType: 'training',
  modelId: null,
  trainingTests: ['distill-stage-b'],
  trainingStage: 'stage_b',
  trainingConfig: {
    distill: {
      enabled: true,
      stage: 'stage_b',
      teacherModelId: 'translategemma-4b-it-wq4k-ef16-hf16',
      studentModelId: 'translategemma-270m-it-wf16-ef16-hf16',
      datasetId: 'en-es',
      languagePair: 'en-es',
    },
  },
  stageAArtifact: '/tmp/distill_stage_a_manifest.json',
  stageAArtifactHash: 'abc123',
  distillArtifactDir: '/tmp/distill',
  teacherModelId: 'translategemma-4b-it-wq4k-ef16-hf16',
  studentModelId: 'translategemma-270m-it-wf16-ef16-hf16',
  distillDatasetId: 'en-es',
  distillLanguagePair: 'en-es',
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
assert.equal(nodeRequest.trainingStage, 'stage_b');
assert.equal(nodeRequest.stageAArtifact, '/tmp/distill_stage_a_manifest.json');
assert.equal(nodeRequest.stageAArtifactHash, 'abc123');
assert.equal(nodeRequest.distillArtifactDir, '/tmp/distill');
assert.equal(nodeRequest.teacherModelId, 'translategemma-4b-it-wq4k-ef16-hf16');
assert.equal(nodeRequest.studentModelId, 'translategemma-270m-it-wf16-ef16-hf16');
assert.equal(nodeRequest.distillDatasetId, 'en-es');
assert.equal(nodeRequest.distillLanguagePair, 'en-es');

console.log('distill-command-threading-parity.test: ok');
