import assert from 'node:assert/strict';

import { runNodeCommand } from '../../src/tooling/node-command-runner.js';
import { runBrowserCommand } from '../../src/tooling/browser-command-runner.js';

async function assertRejectsOnBothSurfaces(request, pattern) {
  await assert.rejects(
    () => runNodeCommand(request),
    pattern
  );
  await assert.rejects(
    () => runBrowserCommand(request),
    pattern
  );
}

await assertRejectsOnBothSurfaces({
  command: 'verify',
  suite: 'training',
  modelId: null,
  trainingStage: 'stage3_unknown',
}, /trainingStage must be one of stage1_joint, stage2_base, stage_a, stage_b/);

await assertRejectsOnBothSurfaces({
  command: 'verify',
  suite: 'inference',
  modelId: 'gemma-3-270m-it-wf16-ef16-hf16',
  trainingTests: ['ul-stage1'],
}, /training-only fields require suite="training" or bench workloadType="training"/);

await assertRejectsOnBothSurfaces({
  command: 'bench',
  workloadType: 'training',
  modelId: null,
  trainingStage: 'stage1_joint',
  trainingSchemaVersion: 2,
}, /trainingSchemaVersion must be 1/);

await assertRejectsOnBothSurfaces({
  command: 'bench',
  workloadType: 'training',
  modelId: null,
  trainingStage: 'stage1_joint',
  trainingBenchSteps: 0,
}, /trainingBenchSteps must be a positive integer/);

await assertRejectsOnBothSurfaces({
  command: 'verify',
  suite: 'training',
  modelId: null,
  trainingStage: 'stage_a',
  forceResumeReason: 'missing flag',
}, /forceResumeReason requires forceResume=true/);

await assert.rejects(
  () => runNodeCommand({
    command: 'lora',
    action: 'compare',
    runRoot: 'reports/training/lora/lora-toy/2026-03-07T00-00-00.000Z',
    runtimePreset: 'modes/debug',
  }),
  /lora does not support runtime input fields on the node operator surface: runtimePreset/
);

await assert.rejects(
  () => runNodeCommand({
    command: 'distill',
    action: 'compare',
    runRoot: 'reports/training/distill/demo/2026-03-07T00-00-00.000Z',
    runtimeConfigUrl: 'https://example.test/runtime.json',
  }),
  /distill does not support runtime input fields on the node operator surface: runtimeConfigUrl/
);

await assert.rejects(
  () => runNodeCommand({
    command: 'lora',
    action: 'compare',
    runRoot: 'reports/training/lora/lora-toy/2026-03-07T00-00-00.000Z',
    runtimeConfig: {
      shared: {
        tooling: {
          intent: 'verify',
        },
      },
    },
  }),
  /lora does not support runtime input fields on the node operator surface: runtimeConfig/
);

console.log('training-command-malformed-surface.test: ok');
