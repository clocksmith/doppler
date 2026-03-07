import assert from 'node:assert/strict';

const { DiagnosticsController } = await import('../../demo/diagnostics-controller.js');

const controller = new DiagnosticsController();

await assert.rejects(
  () => controller.runSuite(null, {
    suite: 'bench',
    modelUrl: 'https://example.com/model',
  }),
  /modelId is required for this suite/
);

await assert.rejects(
  () => controller.runSuite({ modelId: 'toy-model' }, {
    suite: 'bench',
    runtimeConfig: {
      shared: {
        debug: {
          probes: ['decode-latency'],
        },
      },
    },
  }),
  /forbids investigation instrumentation/
);

console.log('diagnostics-controller-contract.test: ok');
