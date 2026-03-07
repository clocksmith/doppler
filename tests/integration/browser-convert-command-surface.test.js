import assert from 'node:assert/strict';

import {
  runBrowserCommand,
  normalizeBrowserCommand,
} from '../../src/tooling/browser-command-runner.js';
import { TOOLING_ENVELOPE_SCHEMA_VERSION } from '../../src/tooling/command-envelope.js';

await assert.rejects(
  () => runBrowserCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelBaseId: 'gemma-3-270m-it-wf16-ef16-hf16',
        },
      },
    },
  }),
  /browser command convert requires options\.convertHandler\(request\) to be provided\./
);

{
  const rawRequest = {
    command: 'convert',
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelBaseId: 'gemma-3-270m-it-wf16-ef16-hf16',
        },
      },
      execution: {
        workers: 4,
      },
    },
  };
  const calls = [];
  const result = await runBrowserCommand(rawRequest, {
    async convertHandler(request) {
      calls.push(request);
      return {
        converted: true,
        outputDir: request.outputDir,
      };
    },
  });

  assert.equal(calls.length, 1);
  assert.equal(calls[0].command, 'convert');
  assert.equal(calls[0].inputDir, '/tmp/in');
  assert.equal(calls[0].outputDir, '/tmp/out');
  assert.equal(calls[0].convertPayload?.execution?.workers, 4);
  assert.equal(result.ok, true);
  assert.equal(result.schemaVersion, TOOLING_ENVELOPE_SCHEMA_VERSION);
  assert.equal(result.surface, 'browser');
  assert.deepEqual(result.request, calls[0]);
  assert.deepEqual(result.result, {
    converted: true,
    outputDir: '/tmp/out',
  });
}

{
  const normalized = normalizeBrowserCommand({
    command: 'bench',
    modelId: 'gemma-3-270m-it-wf16-ef16-hf16',
    cacheMode: 'cold',
  });
  assert.equal(normalized.command, 'bench');
  assert.equal(normalized.suite, 'bench');
  assert.equal(normalized.intent, 'calibrate');
  assert.equal(normalized.modelId, 'gemma-3-270m-it-wf16-ef16-hf16');
  assert.equal(normalized.cacheMode, 'cold');
}

console.log('browser-convert-command-surface.test: ok');
