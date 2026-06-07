import assert from 'node:assert/strict';

import {
  runBrowserCommand,
  normalizeBrowserCommand,
} from '../../src/tooling/browser-command-runner.js';

await assert.rejects(
  () => runBrowserCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelBaseId: 'gemma-3-270m-it-f16-af32',
        },
      },
    },
  }),
  /convert is currently Node-only/
);

await assert.rejects(
  () => runBrowserCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelBaseId: 'gemma-3-270m-it-f16-af32',
        },
      },
    },
  }, null),
  /convert is currently Node-only/
);

await assert.rejects(
  () => runBrowserCommand({
    command: 'convert',
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelBaseId: 'gemma-3-270m-it-f16-af32',
        },
      },
      execution: {
        workers: 4,
      },
    },
  }, {
    async convertHandler() {
      return {
        converted: true,
      };
    },
  }),
  /convert is currently Node-only/
);

{
  const normalized = normalizeBrowserCommand({
    command: 'bench',
    modelId: 'gemma-3-270m-it-f16-af32',
    cacheMode: 'cold',
  });
  assert.equal(normalized.command, 'bench');
  assert.equal(normalized.workload, 'inference');
  assert.equal(normalized.intent, 'calibrate');
  assert.equal(normalized.modelId, 'gemma-3-270m-it-f16-af32');
  assert.equal(normalized.cacheMode, 'cold');
}

console.log('browser-convert-command-surface.test: ok');
