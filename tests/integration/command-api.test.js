import assert from 'node:assert/strict';
import {
  normalizeToolingCommandRequest,
  buildRuntimeContractPatch,
} from '../../src/tooling/command-api.js';

{
  const request = normalizeToolingCommandRequest({
    command: 'test-model',
    suite: 'training',
    modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    trainingTests: ['runner-smoke', 'train-step-metrics'],
    trainingStage: 'stage1_joint',
  });
  assert.equal(request.command, 'test-model');
  assert.equal(request.suite, 'training');
  assert.equal(request.intent, 'verify');
  assert.equal(request.modelId, 'gemma-3-1b-it-wf16-ef16-hf16');
  assert.deepEqual(request.trainingTests, ['runner-smoke', 'train-step-metrics']);
  assert.equal(request.trainingStage, 'stage1_joint');
  assert.equal(request.trainingSchemaVersion, 1);
}

{
  const request = normalizeToolingCommandRequest({
    command: 'test-model',
    suite: 'training',
    modelId: null,
    trainingStage: 'stage1_joint',
  });
  assert.equal(request.modelId, null);
  assert.equal(request.trainingStage, 'stage1_joint');
  assert.equal(request.trainingSchemaVersion, 1);
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'test-model',
      suite: 'bench',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    }),
    /suite must be one of kernels, inference, training, diffusion, energy/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'debug',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
      trainingTests: ['runner-smoke'],
    }),
    /training-only fields require suite="training" or bench workloadType="training"/
  );
}

{
  const request = normalizeToolingCommandRequest({
    command: 'bench',
    modelId: null,
    workloadType: 'training',
    trainingStage: 'stage1_joint',
    trainingTests: ['ul-stage1'],
  });
  assert.equal(request.command, 'bench');
  assert.equal(request.suite, 'bench');
  assert.equal(request.intent, 'calibrate');
  assert.equal(request.workloadType, 'training');
  assert.equal(request.modelId, null);
  assert.equal(request.trainingStage, 'stage1_joint');
  assert.deepEqual(request.trainingTests, ['ul-stage1']);
  assert.equal(request.trainingSchemaVersion, 1);
}

{
  const request = normalizeToolingCommandRequest({
    command: 'bench',
    modelId: null,
    workloadType: 'training',
    trainingStage: 'stage1_joint',
    trainingSchemaVersion: 1,
    trainingBenchSteps: 5,
  });
  assert.equal(request.trainingSchemaVersion, 1);
  assert.equal(request.trainingBenchSteps, 5);
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'test-model',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage1_joint',
      trainingSchemaVersion: 2,
    }),
    /trainingSchemaVersion must be 1/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'bench',
      modelId: null,
      workloadType: 'inference',
      trainingStage: 'stage1_joint',
    }),
    /training-only fields require suite="training" or bench workloadType="training"/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'test-model',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage3_unknown',
    }),
    /trainingStage must be one of stage1_joint, stage2_base, stage_a, stage_b/
  );
}

{
  const request = normalizeToolingCommandRequest({
    command: 'test-model',
    suite: 'training',
    modelId: null,
    trainingStage: 'stage_a',
    teacherModelId: 'translategemma-4b-it-wq4k-ef16-hf16',
    studentModelId: 'translategemma-270m-it-wf16-ef16-hf16',
    distillDatasetId: 'en-es',
    distillDatasetPath: '/tmp/en-es.jsonl',
    distillLanguagePair: 'en-es',
  });
  assert.equal(request.trainingStage, 'stage_a');
  assert.equal(request.teacherModelId, 'translategemma-4b-it-wq4k-ef16-hf16');
  assert.equal(request.studentModelId, 'translategemma-270m-it-wf16-ef16-hf16');
  assert.equal(request.distillDatasetId, 'en-es');
  assert.equal(request.distillDatasetPath, '/tmp/en-es.jsonl');
  assert.equal(request.distillLanguagePair, 'en-es');
}

{
  const patch = buildRuntimeContractPatch({
    command: 'test-model',
    suite: 'training',
    modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
  });
  assert.deepEqual(patch, {
    shared: {
      harness: {
        mode: 'training',
        modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
      },
      tooling: {
        intent: 'verify',
      },
    },
  });
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      convertPayload: null,
    }),
    /convert requires convertPayload\.converterConfig/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      modelId: 'gemma-3-1b-it',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it' },
        },
      },
    }),
    /convert does not accept modelId/
  );
}

{
  const request = normalizeToolingCommandRequest({
    command: 'convert',
    inputDir: '/tmp/model',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
      },
    },
  });
  assert.equal(request.modelId, null);
  assert.equal(request.inputDir, '/tmp/model');
  assert.equal(request.outputDir, '/tmp/out');
  assert.equal(
    request.convertPayload?.converterConfig?.output?.modelBaseId,
    'gemma-3-1b-it-wf16-ef16-hf16'
  );
}

{
  const request = normalizeToolingCommandRequest({
    command: 'convert',
    inputDir: '/tmp/model',
    convertPayload: {
      converterConfig: {
        output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
      },
      execution: {
        workers: 8,
        workerCountPolicy: 'error',
      },
    },
  });
  assert.equal(request.convertPayload?.execution?.workers, 8);
  assert.equal(request.convertPayload?.execution?.workerCountPolicy, 'error');
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
        execution: {
          workers: 0,
        },
      },
    }),
    /convertPayload\.execution\.workers must be a positive integer/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
        execution: {
          workerCountPolicy: 'auto',
        },
      },
    }),
    /workerCountPolicy must be "cap" or "error"/
  );
}

console.log('command-api.test: ok');
