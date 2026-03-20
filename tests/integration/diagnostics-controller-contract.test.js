import assert from 'node:assert/strict';

const { DiagnosticsController } = await import('../../demo/diagnostics-controller.js');

const calls = [];
const controller = new DiagnosticsController({
  async runCommand(request) {
    calls.push(request);
    if (request.workload !== 'kernels' && !request.modelId) {
      throw new Error('modelId is required for this workload. modelUrl is optional when you need an explicit source.');
    }
    if (
      request.command === 'bench'
      && Array.isArray(request.runtimeConfig?.shared?.debug?.probes)
      && request.runtimeConfig.shared.debug.probes.length > 0
    ) {
      throw new Error('calibrate intent forbids investigation instrumentation.');
    }
    return {
      ok: true,
      schemaVersion: 1,
      surface: 'browser',
      request,
      result: {
        suite: request.workload ?? request.command,
        report: { timestamp: '2026-03-08T00:00:00.000Z' },
        metrics: { ok: true },
        reportInfo: null,
      },
    };
  },
});

await assert.rejects(
  () => controller.runSuite(null, {
    workload: 'inference',
    modelUrl: 'https://example.com/model',
  }),
  /modelId is required for this workload/
);

await assert.rejects(
  () => controller.runSuite({ modelId: 'toy-model' }, {
    workload: 'bench',
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

{
  const result = await controller.verifySuite({ modelId: 'toy-model' }, {
    workload: 'inference',
    runtimeProfile: 'profiles/verbose-trace',
    runtimeConfigUrl: 'https://example.test/runtime.json',
    modelUrl: 'https://example.test/model',
  });
  assert.equal(result.suite, 'inference');
  const lastCall = calls.at(-1);
  assert.equal(lastCall.command, 'verify');
  assert.equal(lastCall.workload, 'inference');
  assert.equal(lastCall.configChain, undefined);
  assert.equal(lastCall.runtimeProfile, 'profiles/verbose-trace');
  assert.equal(lastCall.runtimeConfigUrl, 'https://example.test/runtime.json');
  assert.equal(lastCall.modelId, 'toy-model');
}

await assert.rejects(
  () => controller.verifySuite({ modelId: 'toy-model' }, {
    workload: 'inference',
    configChain: ['runtime/base'],
  }),
  /does not accept configChain/
);

console.log('diagnostics-controller-contract.test: ok');
