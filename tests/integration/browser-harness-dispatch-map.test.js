import assert from 'node:assert/strict';
import '../../tools/node-test-runtime-setup.js';

const {
  getBrowserSupportedSuites,
  getBrowserSuiteDispatchMap,
} = await import('../../src/inference/browser-harness.js');
const {
  TOOLING_WORKLOADS,
  TOOLING_VERIFY_WORKLOADS,
} = await import('../../src/tooling/command-api.js');

{
  const suites = getBrowserSupportedSuites();
  assert.deepEqual(suites, [
    'kernels',
    'inference',
    'embedding',
    'training',
    'diffusion',
    'energy',
  ]);
  assert.deepEqual([...TOOLING_WORKLOADS], suites);
}

{
  const dispatchMap = getBrowserSuiteDispatchMap();
  assert.deepEqual(dispatchMap, {
    verify: {
      kernels: 'runKernelSuite',
      inference: 'runInferenceSuite',
      embedding: 'runEmbeddingSuite',
      training: 'runTrainingSuite',
      diffusion: 'runDiffusionSuite',
      energy: 'runEnergySuite',
    },
    debug: {
      inference: 'runInferenceSuite(debug)',
      embedding: 'runEmbeddingSuite(debug)',
    },
    bench: {
      inference: 'runBenchSuite',
      embedding: 'runBenchSuite',
      training: 'runBenchSuite(training)',
      diffusion: 'runBenchSuite(diffusion)',
    },
  });
}

{
  const supported = new Set(getBrowserSupportedSuites());
  for (const workload of TOOLING_VERIFY_WORKLOADS) {
    assert.ok(supported.has(workload), `verify workload "${workload}" must be supported by browser harness`);
  }
}

console.log('browser-harness-dispatch-map.test: ok');
