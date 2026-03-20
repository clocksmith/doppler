import assert from 'node:assert/strict';
import {
  applyRuntimeInputs,
  buildSuiteOptions,
  runWithRuntimeIsolation,
} from '../../src/tooling/command-runner-shared.js';

function createRuntimeBridge(initialRuntime = {}, overlays = {}) {
  let runtime = initialRuntime;
  const calls = [];
  return {
    calls,
    bridge: {
      async applyRuntimeProfile(runtimeProfile, options) {
        calls.push({ type: 'profile', runtimeProfile, options: options || {} });
        runtime = mergeRuntime(runtime, overlays.presets?.[runtimeProfile] ?? null);
      },
      async applyRuntimeConfigFromUrl(runtimeConfigUrl, options) {
        calls.push({ type: 'config-url', runtimeConfigUrl, options: options || {} });
        runtime = mergeRuntime(runtime, overlays.urls?.[runtimeConfigUrl] ?? null);
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

function clone(value) {
  if (typeof structuredClone === 'function') {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value));
}

function mergeRuntime(base, patch) {
  if (!patch) return clone(base);
  const out = clone(base) ?? {};
  for (const [key, value] of Object.entries(patch)) {
    if (
      value &&
      typeof value === 'object' &&
      !Array.isArray(value) &&
      out[key] &&
      typeof out[key] === 'object' &&
      !Array.isArray(out[key])
    ) {
      out[key] = mergeRuntime(out[key], value);
    } else {
      out[key] = clone(value);
    }
  }
  return out;
}

{
  const runtime = createRuntimeBridge({
    shared: {
      harness: { mode: 'bench', modelId: null },
      tooling: { intent: 'calibrate' },
    },
    inference: {
      prompt: 'initial',
      batching: { maxTokens: 16, batchSize: 1 },
    },
  }, {
    presets: {
      'profiles/verbose-trace': {
        inference: {
          prompt: 'preset',
          batching: { maxTokens: 14, batchSize: 2 },
        },
      },
    },
    urls: {
      '/runtime/custom.json': {
        inference: {
          prompt: 'url',
          batching: { batchSize: 6, readbackInterval: 3 },
        },
      },
    },
  });

  await applyRuntimeInputs({
    command: 'verify',
    workload: 'inference',
    intent: 'verify',
    modelId: 'gemma-3-270m-it-f16-af32',
    runtimeProfile: 'profiles/verbose-trace',
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
      type: 'profile',
      runtimeProfile: 'profiles/verbose-trace',
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
  assert.equal(runtime.getRuntime().inference.batching.batchSize, 6);
  assert.equal(runtime.getRuntime().inference.batching.readbackInterval, 3);
  assert.equal(runtime.getRuntime().shared.harness.mode, 'verify');
  assert.equal(runtime.getRuntime().shared.harness.workload, 'inference');
  assert.equal(runtime.getRuntime().shared.harness.modelId, 'gemma-3-270m-it-f16-af32');
  assert.equal(runtime.getRuntime().shared.tooling.intent, 'verify');
}

{
  const runtime = createRuntimeBridge({ inference: { prompt: 'base' } });
  await applyRuntimeInputs({
    command: 'convert',
    workload: null,
    intent: null,
    modelId: null,
    runtimeProfile: null,
    runtimeConfigUrl: null,
    runtimeConfig: null,
    inputDir: '/tmp/in',
    outputDir: '/tmp/out',
    convertPayload: {
      converterConfig: {
        output: {
          modelId: 'gemma-3-270m-it-f16-af32',
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
    workload: 'inference',
    modelId: 'gemma-3-270m-it-f16-af32',
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
    distillSourceLangs: null,
    distillTargetLangs: null,
    distillPairAllowlist: null,
    strictPairContract: null,
    trainingSchemaVersion: null,
    trainingBenchSteps: null,
    checkpointEvery: null,
    modelUrl: null,
    runtimeProfile: null,
    captureOutput: true,
    keepPipeline: false,
    report: null,
    timestamp: '2026-02-22T00:00:00.000Z',
    searchParams: null,
  }, 'node');

  assert.deepEqual(suiteOptions, {
    mode: 'verify',
    workload: 'inference',
    command: 'verify',
    surface: 'node',
    modelId: 'gemma-3-270m-it-f16-af32',
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
    distillSourceLangs: undefined,
    distillTargetLangs: undefined,
    distillPairAllowlist: undefined,
    strictPairContract: undefined,
    distillShardIndex: undefined,
    distillShardCount: undefined,
    resumeFrom: undefined,
    forceResume: undefined,
    forceResumeReason: undefined,
    forceResumeSource: undefined,
    checkpointOperator: undefined,
    trainingSchemaVersion: undefined,
    trainingBenchSteps: undefined,
    checkpointEvery: undefined,
    modelUrl: undefined,
    cacheMode: 'warm',
    loadMode: null,
    runtimeProfile: null,
    captureOutput: true,
    keepPipeline: false,
    report: undefined,
    timestamp: '2026-02-22T00:00:00.000Z',
    searchParams: undefined,
  });
}

{
  const suiteOptions = buildSuiteOptions({
    command: 'bench',
    workload: 'embedding',
    modelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
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
    distillSourceLangs: null,
    distillTargetLangs: null,
    distillPairAllowlist: null,
    strictPairContract: null,
    trainingSchemaVersion: null,
    trainingBenchSteps: null,
    checkpointEvery: null,
    modelUrl: null,
    runtimeProfile: 'profiles/vector-throughput',
    captureOutput: false,
    keepPipeline: false,
    report: null,
    timestamp: null,
    searchParams: null,
  }, 'browser');

  assert.equal(suiteOptions.mode, 'bench');
  assert.equal(suiteOptions.workload, 'embedding');
  assert.equal(suiteOptions.command, 'bench');
  assert.equal(suiteOptions.expectedModelType, 'embedding');
  assert.equal(suiteOptions.runtimeProfile, 'profiles/vector-throughput');
}

{
  const trainingSuiteOptions = buildSuiteOptions({
    command: 'verify',
    workload: 'training',
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
    distillSourceLangs: ['en'],
    distillTargetLangs: ['es'],
    distillPairAllowlist: ['en-es'],
    strictPairContract: true,
    forceResume: true,
    forceResumeReason: 'intentional migration',
    forceResumeSource: 'verify:node',
    checkpointOperator: 'qa-bot',
    trainingSchemaVersion: 1,
    trainingBenchSteps: 4,
    checkpointEvery: 2,
    modelUrl: null,
    runtimeProfile: null,
    captureOutput: false,
    keepPipeline: false,
    report: null,
    timestamp: null,
    searchParams: null,
  }, 'node');

  assert.deepEqual(trainingSuiteOptions, {
    mode: 'verify',
    workload: 'training',
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
    distillSourceLangs: ['en'],
    distillTargetLangs: ['es'],
    distillPairAllowlist: ['en-es'],
    strictPairContract: true,
    distillShardIndex: undefined,
    distillShardCount: undefined,
    resumeFrom: undefined,
    forceResume: true,
    forceResumeReason: 'intentional migration',
    forceResumeSource: 'verify:node',
    checkpointOperator: 'qa-bot',
    trainingSchemaVersion: 1,
    trainingBenchSteps: 4,
    checkpointEvery: 2,
    modelUrl: undefined,
    cacheMode: 'warm',
    loadMode: null,
    runtimeProfile: null,
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
    async applyRuntimeProfile() {},
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
  const runtime = createRuntimeBridge({
    shared: {
      debug: {
        profiler: {
          enabled: true,
        },
      },
    },
  });

  await assert.rejects(
    () => applyRuntimeInputs({
      command: 'bench',
      workload: 'inference',
      intent: 'calibrate',
      modelId: 'gemma-3-270m-it-f16-af32',
      runtimeConfig: {
        shared: {
          benchmark: {
            run: {
              profile: true,
            },
          },
        },
      },
    }, runtime.bridge),
    /tooling command: calibrate intent forbids investigation instrumentation/
  );
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
