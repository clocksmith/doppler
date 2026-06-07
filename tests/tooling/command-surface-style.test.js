import assert from 'node:assert/strict';

import {
  ensureCommandSupportedOnSurface,
  normalizeToolingCommandRequest,
} from '../../src/tooling/command-api.js';
import { runBrowserCommand } from '../../src/tooling/browser-command-runner.js';
import {
  applyRuntimeInputs,
  buildSuiteOptions,
} from '../../src/tooling/command-runner-shared.js';

const convertRequest = {
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
};

{
  const request = normalizeToolingCommandRequest({
    command: 'verify',
    workload: 'inference',
    modelId: 'gemma-3-270m-it-f16-af32',
    configChain: [' profiles/base ', 'profiles/override'],
  });

  assert.deepEqual(request.configChain, ['profiles/base', 'profiles/override']);
}

{
  assert.throws(
    () => normalizeToolingCommandRequest({
      ...convertRequest,
      configChain: ['profiles/base'],
    }),
    /convert does not accept configChain/
  );

  assert.throws(
    () => normalizeToolingCommandRequest({
      command: 'lora',
      action: 'run',
      workloadPath: '/tmp/workload.json',
      configChain: ['profiles/base'],
    }),
    /lora does not accept configChain/
  );
}

{
  assert.throws(
    () => ensureCommandSupportedOnSurface(convertRequest, 'browser'),
    /convert is currently Node-only/
  );

  await assert.rejects(
    () => runBrowserCommand(convertRequest),
    /convert is currently Node-only/
  );
}

{
  const suiteOptions = buildSuiteOptions({
    command: 'verify',
    workload: 'inference',
    modelId: 'gemma-3-270m-it-f16-af32',
    cacheMode: null,
    captureOutput: false,
    keepPipeline: false,
  }, 'node');

  assert.equal(suiteOptions.cacheMode, null);

  const coldSuiteOptions = buildSuiteOptions({
    command: 'bench',
    workload: 'inference',
    modelId: 'gemma-3-270m-it-f16-af32',
    cacheMode: 'cold',
    captureOutput: false,
    keepPipeline: false,
  }, 'browser');

  assert.equal(coldSuiteOptions.cacheMode, 'cold');
}

{
  const calls = [];
  let runtime = null;
  const bridge = {
    async loadRuntimeConfigFromRef(ref) {
      calls.push(`chain:${ref}`);
      return {
        runtime: {
          shared: {
            chainRef: ref,
          },
        },
      };
    },
    async applyRuntimeProfile(runtimeProfile) {
      calls.push(`profile:${runtimeProfile}`);
    },
    async applyRuntimeConfigFromUrl(runtimeConfigUrl) {
      calls.push(`url:${runtimeConfigUrl}`);
    },
    getRuntimeConfig() {
      return runtime;
    },
    setRuntimeConfig(nextRuntime) {
      runtime = nextRuntime;
    },
    resetRuntimeConfig() {
      runtime = null;
    },
  };

  await applyRuntimeInputs({
    command: 'verify',
    workload: 'inference',
    intent: 'verify',
    modelId: 'gemma-3-270m-it-f16-af32',
    configChain: ['profiles/base', 'profiles/override'],
    runtimeProfile: 'profiles/final',
    runtimeConfigUrl: '/runtime/final.json',
    runtimeConfig: {
      shared: {
        finalOverride: true,
      },
    },
  }, bridge);

  assert.deepEqual(calls, [
    'chain:profiles/base',
    'chain:profiles/override',
    'profile:profiles/final',
    'url:/runtime/final.json',
  ]);
  assert.equal(runtime.shared.chainRef, 'profiles/override');
  assert.equal(runtime.shared.finalOverride, true);
}

console.log('command-surface-style.test: ok');
