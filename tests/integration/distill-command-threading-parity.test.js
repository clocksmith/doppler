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
      teacherModelId: 'translategemma-4b-it-q4k-ehf16-af32',
      studentModelId: 'translategemma-270m-it-f16-af32',
      datasetId: 'en-es',
      languagePair: 'en-es',
    },
  },
  stageAArtifact: '/tmp/distill_stage_a_manifest.json',
  stageAArtifactHash: 'abc123',
  distillArtifactDir: '/tmp/distill',
  teacherModelId: 'translategemma-4b-it-q4k-ehf16-af32',
  studentModelId: 'translategemma-270m-it-f16-af32',
  distillDatasetId: 'en-es',
  distillDatasetPath: '/tmp/translate_distill_pairs_en_es.jsonl',
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
assert.equal(nodeRequest.teacherModelId, 'translategemma-4b-it-q4k-ehf16-af32');
assert.equal(nodeRequest.studentModelId, 'translategemma-270m-it-f16-af32');
assert.equal(nodeRequest.distillDatasetId, 'en-es');
assert.equal(nodeRequest.distillDatasetPath, '/tmp/translate_distill_pairs_en_es.jsonl');
assert.equal(nodeRequest.distillLanguagePair, 'en-es');

console.log('distill-command-threading-parity.test: ok');
