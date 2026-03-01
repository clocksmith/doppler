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
  });
  assert.equal(request.command, 'test-model');
  assert.equal(request.suite, 'training');
  assert.equal(request.intent, 'verify');
  assert.equal(request.modelId, 'gemma-3-1b-it-wf16-ef16-hf16');
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
