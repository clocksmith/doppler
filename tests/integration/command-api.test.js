import assert from 'node:assert/strict';
import { normalizeToolingCommandRequest } from '../../src/tooling/command-api.js';

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
          output: { modelId: 'gemma-3-1b-it' },
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
        output: { modelId: 'gemma-3-1b-it-f16-f32a-wf16' },
      },
    },
  });
  assert.equal(request.modelId, null);
  assert.equal(request.inputDir, '/tmp/model');
  assert.equal(request.outputDir, '/tmp/out');
  assert.equal(
    request.convertPayload?.converterConfig?.output?.modelId,
    'gemma-3-1b-it-f16-f32a-wf16'
  );
}

console.log('command-api.test: ok');
