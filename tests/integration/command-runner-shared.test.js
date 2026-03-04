import assert from 'node:assert/strict';
import {
  applyRuntimeInputs,
  buildSuiteOptions,
  runWithRuntimeIsolation,
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
    command: 'verify',
    suite: 'inference',
    intent: 'verify',
    modelId: 'gemma-3-270m-it-wf16-ef16-hf16',
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
  assert.equal(runtime.getRuntime().shared.harness.modelId, 'gemma-3-270m-it-wf16-ef16-hf16');
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
          modelId: 'gemma-3-270m-it-wf16-ef16-hf16',
        },
      },
    },
  }, runtime.bridge);

  assert.deepEqual(runtime.calls, []);
  assert.equal(runtime.getRuntime(), null);
}

{
  const suiteOptions = buildSuiteOptions({
    command: 'verify',
    suite: 'inference',
    modelId: 'gemma-3-270m-it-wf16-ef16-hf16',
    workloadType: null,
    trainingTests: null,
    trainingStage: null,
    trainingConfig: null,
    stage1Artifact: null,
    stage1ArtifactHash: null,
    ulArtifactDir: null,
    stageAArtifact: null,
    stageAArtifactHash: null,
    distillArtifactDir: null,
    teacherModelId: null,
    studentModelId: null,
    distillDatasetId: null,
    distillDatasetPath: null,
    distillLanguagePair: null,
    trainingSchemaVersion: null,
    trainingBenchSteps: null,
    modelUrl: null,
    runtimePreset: null,
    captureOutput: true,
    keepPipeline: false,
    report: null,
    timestamp: '2026-02-22T00:00:00.000Z',
    searchParams: null,
  }, 'node');

  assert.deepEqual(suiteOptions, {
    suite: 'inference',
    command: 'verify',
    surface: 'node',
    modelId: 'gemma-3-270m-it-wf16-ef16-hf16',
    workloadType: undefined,
    trainingTests: undefined,
    trainingStage: undefined,
    trainingConfig: undefined,
    stage1Artifact: undefined,
    stage1ArtifactHash: undefined,
    ulArtifactDir: undefined,
    stageAArtifact: undefined,
    stageAArtifactHash: undefined,
    distillArtifactDir: undefined,
    teacherModelId: undefined,
    studentModelId: undefined,
    distillDatasetId: undefined,
    distillDatasetPath: undefined,
    distillLanguagePair: undefined,
    distillShardIndex: undefined,
    distillShardCount: undefined,
    resumeFrom: undefined,
    forceResume: undefined,
    forceResumeReason: undefined,
    forceResumeSource: undefined,
    checkpointOperator: undefined,
    trainingSchemaVersion: undefined,
    trainingBenchSteps: undefined,
    modelUrl: undefined,
    cacheMode: 'warm',
    loadMode: null,
    runtimePreset: null,
    captureOutput: true,
    keepPipeline: false,
    report: undefined,
    timestamp: '2026-02-22T00:00:00.000Z',
    searchParams: undefined,
  });
}

{
  const trainingSuiteOptions = buildSuiteOptions({
    command: 'verify',
    suite: 'training',
    modelId: null,
    workloadType: 'training',
    trainingTests: ['ul-stage1'],
    trainingStage: 'stage1_joint',
    trainingConfig: { ul: { enabled: true, stage: 'stage1_joint' } },
    stage1Artifact: '/tmp/stage1.json',
    stage1ArtifactHash: 'abc123',
    ulArtifactDir: '/tmp/ul',
    stageAArtifact: null,
    stageAArtifactHash: null,
    distillArtifactDir: null,
    teacherModelId: null,
    studentModelId: null,
    distillDatasetId: null,
    distillDatasetPath: null,
    distillLanguagePair: null,
    forceResume: true,
    forceResumeReason: 'intentional migration',
    forceResumeSource: 'verify:node',
    checkpointOperator: 'qa-bot',
    trainingSchemaVersion: 1,
    trainingBenchSteps: 4,
    modelUrl: null,
    runtimePreset: null,
    captureOutput: false,
    keepPipeline: false,
    report: null,
    timestamp: null,
    searchParams: null,
  }, 'node');

  assert.deepEqual(trainingSuiteOptions, {
    suite: 'training',
    command: 'verify',
    surface: 'node',
    modelId: undefined,
    workloadType: 'training',
    trainingTests: ['ul-stage1'],
    trainingStage: 'stage1_joint',
    trainingConfig: { ul: { enabled: true, stage: 'stage1_joint' } },
    stage1Artifact: '/tmp/stage1.json',
    stage1ArtifactHash: 'abc123',
    ulArtifactDir: '/tmp/ul',
    stageAArtifact: undefined,
    stageAArtifactHash: undefined,
    distillArtifactDir: undefined,
    teacherModelId: undefined,
    studentModelId: undefined,
    distillDatasetId: undefined,
    distillDatasetPath: undefined,
    distillLanguagePair: undefined,
    distillShardIndex: undefined,
    distillShardCount: undefined,
    resumeFrom: undefined,
    forceResume: true,
    forceResumeReason: 'intentional migration',
    forceResumeSource: 'verify:node',
    checkpointOperator: 'qa-bot',
    trainingSchemaVersion: 1,
    trainingBenchSteps: 4,
    modelUrl: undefined,
    cacheMode: 'warm',
    loadMode: null,
    runtimePreset: null,
    captureOutput: false,
    keepPipeline: false,
    report: undefined,
    timestamp: undefined,
    searchParams: undefined,
  });
}

{
  let runtime = {
    inference: {
      prompt: 'baseline',
    },
  };
  let kernelPath = '/tmp/kernel/default.wgsl';
  let kernelSource = 'runtime';
  let kernelPolicy = { policy: 'strict' };
  const bridge = {
    getRuntimeConfig() {
      return runtime;
    },
    setRuntimeConfig(nextRuntime) {
      runtime = nextRuntime;
    },
    getActiveKernelPath() {
      return kernelPath;
    },
    getActiveKernelPathSource() {
      return kernelSource;
    },
    getActiveKernelPathPolicy() {
      return kernelPolicy;
    },
    setActiveKernelPath(nextKernelPath, nextKernelSource, nextKernelPolicy) {
      kernelPath = nextKernelPath;
      kernelSource = nextKernelSource;
      kernelPolicy = nextKernelPolicy;
    },
  };

  const result = await runWithRuntimeIsolation(bridge, async () => {
    bridge.setRuntimeConfig({
      inference: {
        prompt: 'mutated',
      },
    });
    bridge.setActiveKernelPath('/tmp/kernel/override.wgsl', 'override', { policy: 'override' });
    return 'ok';
  });

  assert.equal(result, 'ok');
  assert.deepEqual(runtime, {
    inference: {
      prompt: 'baseline',
    },
  });
  assert.equal(kernelPath, '/tmp/kernel/default.wgsl');
  assert.equal(kernelSource, 'runtime');
  assert.deepEqual(kernelPolicy, { policy: 'strict' });
}

{
  let runtime = {
    shared: {
      tooling: {
        intent: 'verify',
      },
    },
  };
  let kernelPath = null;
  let kernelSource = 'none';
  let kernelPolicy = { mode: 'auto' };
  const bridge = {
    getRuntimeConfig() {
      return runtime;
    },
    setRuntimeConfig(nextRuntime) {
      runtime = nextRuntime;
    },
    getActiveKernelPath() {
      return kernelPath;
    },
    getActiveKernelPathSource() {
      return kernelSource;
    },
    getActiveKernelPathPolicy() {
      return kernelPolicy;
    },
    setActiveKernelPath(nextKernelPath, nextKernelSource, nextKernelPolicy) {
      kernelPath = nextKernelPath;
      kernelSource = nextKernelSource;
      kernelPolicy = nextKernelPolicy;
    },
  };

  await assert.rejects(
    () => runWithRuntimeIsolation(bridge, async () => {
      bridge.setRuntimeConfig({
        shared: {
          tooling: {
            intent: 'mutated',
          },
        },
      });
      bridge.setActiveKernelPath('/tmp/kernel/override.wgsl', 'override', { mode: 'manual' });
      throw new Error('isolation failure');
    }),
    /isolation failure/
  );

  assert.deepEqual(runtime, {
    shared: {
      tooling: {
        intent: 'verify',
      },
    },
  });
  assert.equal(kernelPath, null);
  assert.equal(kernelSource, 'none');
  assert.deepEqual(kernelPolicy, { mode: 'auto' });
}

{
  let resetCalls = 0;
  let setCalls = 0;
  await applyRuntimeInputs({
    command: 'convert',
    inputDir: '/tmp/input',
    convertPayload: {
      converterConfig: {},
    },
  }, {
    async applyRuntimePreset() {},
    async applyRuntimeConfigFromUrl() {},
    getRuntimeConfig() {
      return {
        inference: {
          prompt: 'base',
        },
      };
    },
    setRuntimeConfig() {
      setCalls += 1;
    },
    resetRuntimeConfig() {
      resetCalls += 1;
    },
  });

  assert.equal(resetCalls, 1);
  assert.equal(setCalls, 0);
}

{
  await assert.rejects(
    () => applyRuntimeInputs({
      command: 'convert',
      inputDir: '/tmp/input',
      convertPayload: {
        converterConfig: {},
      },
    }, {
      getRuntimeConfig() {
        return null;
      },
    }),
    /runtime bridge must provide setRuntimeConfig\(\)\./
  );
}

console.log('command-runner-shared.test: ok');
