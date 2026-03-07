import assert from 'node:assert/strict';
import {
  normalizeToolingCommandRequest,
  buildRuntimeContractPatch,
  ensureCommandSupportedOnSurface,
} from '../../src/tooling/command-api.js';

{
  const request = normalizeToolingCommandRequest({
    command: 'verify',
    suite: 'training',
    modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    trainingTests: ['runner-smoke', 'train-step-metrics'],
    trainingStage: 'stage1_joint',
  });
  assert.equal(request.command, 'verify');
  assert.equal(request.suite, 'training');
  assert.equal(request.intent, 'verify');
  assert.equal(request.modelId, 'gemma-3-1b-it-wf16-ef16-hf16');
  assert.deepEqual(request.trainingTests, ['runner-smoke', 'train-step-metrics']);
  assert.equal(request.trainingStage, 'stage1_joint');
  assert.equal(request.trainingSchemaVersion, 1);
}

{
  const request = normalizeToolingCommandRequest({
    command: 'verify',
    suite: 'training',
    modelId: null,
    trainingStage: 'stage_a',
    forceResume: true,
    forceResumeReason: 'intentional schema transition',
    forceResumeSource: 'verify:node',
    checkpointOperator: 'release-engineer',
  });
  assert.equal(request.forceResume, true);
  assert.equal(request.forceResumeReason, 'intentional schema transition');
  assert.equal(request.forceResumeSource, 'verify:node');
  assert.equal(request.checkpointOperator, 'release-engineer');
}

{
  const request = normalizeToolingCommandRequest({
    command: 'verify',
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
      command: 'verify',
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
    checkpointEvery: 25,
    distillSourceLangs: ['en'],
    distillTargetLangs: ['es'],
    distillPairAllowlist: ['en-es', 'es-en'],
    strictPairContract: true,
  });
  assert.equal(request.trainingSchemaVersion, 1);
  assert.equal(request.trainingBenchSteps, 5);
  assert.equal(request.checkpointEvery, 25);
  assert.deepEqual(request.distillSourceLangs, ['en']);
  assert.deepEqual(request.distillTargetLangs, ['es']);
  assert.deepEqual(request.distillPairAllowlist, ['en-es', 'es-en']);
  assert.equal(request.strictPairContract, true);
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'bench',
      suite: 'training',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    }),
    /requires suite "bench" and does not accept "training"/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
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
      command: 'verify',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage_a',
      checkpointEvery: 0,
    }),
    /checkpointEvery must be a positive integer/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage_a',
      distillSourceLangs: 'en,es',
    }),
    /distillSourceLangs must be an array of strings/
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
      command: 'verify',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage_a',
      forceResumeReason: 'missing flag',
    }),
    /forceResumeReason requires forceResume=true/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage_a',
      forceResumeSource: 'verify:node',
    }),
    /forceResumeSource requires forceResume=true/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage_a',
      checkpointOperator: 'release-engineer',
    }),
    /checkpointOperator requires forceResume=true/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'training',
      modelId: null,
      trainingStage: 'stage3_unknown',
    }),
    /trainingStage must be one of stage1_joint, stage2_base, stage_a, stage_b/
  );
}

{
  const request = normalizeToolingCommandRequest({
    command: 'verify',
    suite: 'training',
    modelId: null,
    trainingStage: 'stage_a',
    teacherModelId: 'translategemma-4b-it-wq4k-ef16-hf16',
    studentModelId: 'translategemma-270m-it-wf16-ef16-hf16',
    distillDatasetId: 'en-es',
    distillDatasetPath: '/tmp/en-es.jsonl',
    distillLanguagePair: 'en-es',
    distillShardIndex: 2,
    distillShardCount: 5,
    resumeFrom: '/tmp/checkpoint-latest.json',
  });
  assert.equal(request.trainingStage, 'stage_a');
  assert.equal(request.teacherModelId, 'translategemma-4b-it-wq4k-ef16-hf16');
  assert.equal(request.studentModelId, 'translategemma-270m-it-wf16-ef16-hf16');
  assert.equal(request.distillDatasetId, 'en-es');
  assert.equal(request.distillDatasetPath, '/tmp/en-es.jsonl');
  assert.equal(request.distillLanguagePair, 'en-es');
  assert.equal(request.distillShardIndex, 2);
  assert.equal(request.distillShardCount, 5);
  assert.equal(request.resumeFrom, '/tmp/checkpoint-latest.json');
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'bench',
      workloadType: 'training',
      trainingStage: 'stage_a',
      distillShardIndex: 6,
      distillShardCount: 5,
    }),
    /distillShardIndex must be <= distillShardCount/
  );
}

{
  const patch = buildRuntimeContractPatch({
    command: 'verify',
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
  const normalized = normalizeToolingCommandRequest({
    command: 'verify',
    suite: 'inference',
    modelId: 'gemma-3-270m-it-wq4k-ef16-hf16',
  });
  const patch = buildRuntimeContractPatch(normalized);
  assert.deepEqual(patch, {
    shared: {
      harness: {
        mode: 'inference',
        modelId: 'gemma-3-270m-it-wq4k-ef16-hf16',
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
        rowChunkRows: 512,
        rowChunkMinTensorBytes: 33554432,
        maxInFlightJobs: 24,
      },
    },
  });
  assert.equal(request.convertPayload?.execution?.workers, 8);
  assert.equal(request.convertPayload?.execution?.workerCountPolicy, 'error');
  assert.equal(request.convertPayload?.execution?.rowChunkRows, 512);
  assert.equal(request.convertPayload?.execution?.rowChunkMinTensorBytes, 33554432);
  assert.equal(request.convertPayload?.execution?.maxInFlightJobs, 24);
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      runtimePreset: 'debug',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
      },
    }),
    /convert does not accept runtimePreset/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      runtimeConfigUrl: '/tmp/runtime.json',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
      },
    }),
    /convert does not accept runtimeConfigUrl/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      runtimeConfig: {
        inference: {
          prompt: 'hello',
        },
      },
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
      },
    }),
    /convert does not accept runtimeConfig/
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
          rowChunkRows: 0,
        },
      },
    }),
    /convertPayload\.execution\.rowChunkRows must be a positive integer/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest(null),
    /request must be an object/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({}),
    /command is required/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({ command: 'train' }),
    /unsupported command/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    }),
    /suite is required/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'unknown',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    }),
    /unsupported suite/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'inference',
      modelId: null,
    }),
    /modelId is required/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'bench',
      workloadType: 'training',
      trainingStage: 'stage1_joint',
      trainingTests: 'runner-smoke',
    }),
    /trainingTests must be an array of strings/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'bench',
      workloadType: 'training',
      modelId: null,
      trainingTests: ['runner-smoke', 7],
      trainingStage: 'stage1_joint',
    }),
    /trainingTests\[1\] must be a string/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'bench',
      workloadType: 'training',
      modelId: null,
      trainingTests: ['runner-smoke', ''],
      trainingStage: 'stage1_joint',
    }),
    /trainingTests\[1\] must not be empty/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'inference',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
      captureOutput: 'true',
    }),
    /captureOutput must be a boolean/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'verify',
      suite: 'inference',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
      runtimePreset: 7,
    }),
    /runtimePreset must be a string/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'debug',
      suite: 'inference',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    }),
    /requires suite "debug" and does not accept "inference"/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      cacheMode: 'hot',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
      },
    }),
    /cacheMode must be "cold" or "warm"/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      loadMode: 'disk',
      convertPayload: {
        converterConfig: {
          output: { modelBaseId: 'gemma-3-1b-it-wf16-ef16-hf16' },
        },
      },
    }),
    /loadMode must be "opfs", "http", or "memory"/
  );
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'convert',
      inputDir: '/tmp/model',
      convertPayload: {},
    }),
    /convert requires convertPayload\.converterConfig/
  );
}

{
  assert.throws(
    () => ensureCommandSupportedOnSurface({
      command: 'verify',
      suite: 'inference',
      modelId: 'gemma-3-1b-it-wf16-ef16-hf16',
    }, 'desktop'),
    /unsupported surface/
  );
}

console.log('command-api.test: ok');
