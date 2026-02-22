import assert from 'node:assert/strict';
import {
  applyRuntimeInputs,
  buildSuiteOptions,
} from '../../src/tooling/command-runner-shared.js';

function createRuntimeBridge(initialRuntime = {}) {
  let runtime = initialRuntime;
  const calls = [];
  return {
    calls,
    bridge: {
      async applyRuntimePreset(runtimePreset, options) {
        calls.push({ type: 'preset', runtimePreset, options: options || {} });
      },
      async applyRuntimeConfigFromUrl(runtimeConfigUrl, options) {
        calls.push({ type: 'config-url', runtimeConfigUrl, options: options || {} });
      },
      getRuntimeConfig() {
        return runtime;
      },
      setRuntimeConfig(nextRuntime) {
        runtime = nextRuntime;
      },
    },
    getRuntime() {
      return runtime;
    },
  };
}

{
  const runtime = createRuntimeBridge({
    shared: {
      harness: { mode: 'bench', modelId: null },
      tooling: { intent: 'calibrate' },
    },
    inference: {
      prompt: 'initial',
      batching: { maxTokens: 16 },
    },
  });

  await applyRuntimeInputs({
    command: 'test-model',
    suite: 'inference',
    intent: 'verify',
    modelId: 'gemma-3-270m-it-f16-f32a-wf16',
    runtimePreset: 'modes/debug',
    runtimeConfigUrl: '/runtime/custom.json',
    runtimeConfig: {
      inference: {
        prompt: 'The sky is',
        batching: { maxTokens: 8 },
      },
    },
  }, runtime.bridge, { source: 'test' });

  assert.deepEqual(runtime.calls, [
    {
      type: 'preset',
      runtimePreset: 'modes/debug',
      options: { source: 'test' },
    },
    {
      type: 'config-url',
      runtimeConfigUrl: '/runtime/custom.json',
      options: { source: 'test' },
    },
  ]);

  assert.equal(runtime.getRuntime().inference.prompt, 'The sky is');
  assert.equal(runtime.getRuntime().inference.batching.maxTokens, 8);
  assert.equal(runtime.getRuntime().shared.harness.mode, 'inference');
  assert.equal(runtime.getRuntime().shared.harness.modelId, 'gemma-3-270m-it-f16-f32a-wf16');
  assert.equal(runtime.getRuntime().shared.tooling.intent, 'verify');
}

{
  const runtime = createRuntimeBridge({ inference: { prompt: 'base' } });
  await applyRuntimeInputs({
    command: 'convert',
    suite: null,
    intent: null,
    modelId: null,
    runtimePreset: null,
    runtimeConfigUrl: null,
    runtimeConfig: null,
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelId: 'gemma-3-270m-it-f16-f32a-wf16',
        },
      },
    },
  }, runtime.bridge);

  assert.deepEqual(runtime.calls, []);
  assert.deepEqual(runtime.getRuntime(), { inference: { prompt: 'base' } });
}

{
  const suiteOptions = buildSuiteOptions({
    suite: 'inference',
    modelId: 'gemma-3-270m-it-f16-f32a-wf16',
    modelUrl: null,
    runtimePreset: null,
    captureOutput: true,
    keepPipeline: false,
    report: null,
    timestamp: '2026-02-22T00:00:00.000Z',
    searchParams: null,
  });

  assert.deepEqual(suiteOptions, {
    suite: 'inference',
    modelId: 'gemma-3-270m-it-f16-f32a-wf16',
    modelUrl: undefined,
    runtimePreset: null,
    captureOutput: true,
    keepPipeline: false,
    report: undefined,
    timestamp: '2026-02-22T00:00:00.000Z',
    searchParams: undefined,
  });
}

console.log('command-runner-shared.test: ok');
