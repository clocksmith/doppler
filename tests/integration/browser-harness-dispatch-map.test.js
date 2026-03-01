import assert from 'node:assert/strict';

const {
  getBrowserSupportedSuites,
  getBrowserSuiteDispatchMap,
} = await import('../../src/inference/browser-harness.js');
const {
  TOOLING_SUITES,
  TOOLING_VERIFY_SUITES,
} = await import('../../src/tooling/command-api.js');

{
  const suites = getBrowserSupportedSuites();
  assert.deepEqual(suites, [
    'kernels',
    'inference',
    'training',
    'bench',
    'debug',
    'diffusion',
    'energy',
  ]);
  assert.deepEqual([...TOOLING_SUITES], suites);
}

{
  const dispatchMap = getBrowserSuiteDispatchMap();
  assert.deepEqual(dispatchMap, {
    kernels: 'runKernelSuite',
    inference: 'runInferenceSuite',
    training: 'runTrainingSuite',
    bench: 'runBenchSuite',
    debug: 'runInferenceSuite(debug)',
    diffusion: 'runDiffusionSuite',
    energy: 'runEnergySuite',
  });
}

{
  const supported = new Set(getBrowserSupportedSuites());
  for (const suite of TOOLING_VERIFY_SUITES) {
    assert.ok(supported.has(suite), `verify suite "${suite}" must be supported by browser harness`);
  }
}

console.log('browser-harness-dispatch-map.test: ok');
