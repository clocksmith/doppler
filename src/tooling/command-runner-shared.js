import { buildRuntimeContractPatch } from './command-api.js';
import { applyOrderedRuntimeInputs } from './runtime-input-composition.js';

function cloneRuntimeConfig(runtimeConfig) {
  if (runtimeConfig == null) return runtimeConfig;
  if (typeof structuredClone === 'function') {
    return structuredClone(runtimeConfig);
  }
  return JSON.parse(JSON.stringify(runtimeConfig));
}

function assertCalibrateRuntimeCompatibility(request, runtimeConfig) {
  if (request?.intent !== 'calibrate') {
    return;
  }

  const shared = runtimeConfig?.shared ?? {};
  const debug = shared.debug ?? {};
  const benchmarkRun = shared.benchmark?.run ?? {};
  const violations = [];

  if (debug.trace?.enabled === true) {
    violations.push('runtime.shared.debug.trace.enabled');
  }
  if (debug.pipeline?.enabled === true) {
    violations.push('runtime.shared.debug.pipeline.enabled');
  }
  if (Array.isArray(debug.probes) && debug.probes.length > 0) {
    violations.push('runtime.shared.debug.probes');
  }
  if (debug.profiler?.enabled === true) {
    violations.push('runtime.shared.debug.profiler.enabled');
  }
  if (benchmarkRun.debug === true) {
    violations.push('runtime.shared.benchmark.run.debug');
  }
  if (benchmarkRun.profile === true) {
    violations.push('runtime.shared.benchmark.run.profile');
  }
  if (benchmarkRun.captureMemoryTimeSeries === true) {
    violations.push('runtime.shared.benchmark.run.captureMemoryTimeSeries');
  }

  if (violations.length > 0) {
    throw new Error(
      `tooling command: calibrate intent forbids investigation instrumentation (${violations.join(', ')}). ` +
      'Disable those runtime config fields or use the debug command instead.'
    );
  }
}

function resetRuntimeState(runtimeBridge) {
  if (!runtimeBridge?.setRuntimeConfig) {
    throw new Error('runtime bridge must provide setRuntimeConfig().');
  }

  if (typeof runtimeBridge.resetRuntimeConfig === 'function') {
    runtimeBridge.resetRuntimeConfig();
    return;
  }

  runtimeBridge.setRuntimeConfig(null);
}

function snapshotRuntimeState(runtimeBridge) {
  return {
    runtimeConfig: cloneRuntimeConfig(runtimeBridge.getRuntimeConfig()),
    activeKernelPath: runtimeBridge.getActiveKernelPath
      ? runtimeBridge.getActiveKernelPath()
      : null,
    activeKernelPathSource: runtimeBridge.getActiveKernelPathSource
      ? runtimeBridge.getActiveKernelPathSource()
      : 'none',
    activeKernelPathPolicy: runtimeBridge.getActiveKernelPathPolicy
      ? runtimeBridge.getActiveKernelPathPolicy()
      : null,
  };
}

function restoreRuntimeState(runtimeBridge, snapshot) {
  if (!snapshot) {
    return;
  }

  if (snapshot.runtimeConfig != null) {
    runtimeBridge.setRuntimeConfig(snapshot.runtimeConfig);
  } else {
    resetRuntimeState(runtimeBridge);
  }

  if (
    snapshot.activeKernelPath !== null
    && typeof runtimeBridge.setActiveKernelPath === 'function'
  ) {
    runtimeBridge.setActiveKernelPath(
      snapshot.activeKernelPath,
      snapshot.activeKernelPathSource,
      snapshot.activeKernelPathPolicy
    );
    return;
  }

  if (typeof runtimeBridge.setActiveKernelPath === 'function') {
    runtimeBridge.setActiveKernelPath(null, 'none', snapshot.activeKernelPathPolicy);
  }
}

export async function applyRuntimeInputs(request, runtimeBridge, options = {}) {
  resetRuntimeState(runtimeBridge);
  await applyOrderedRuntimeInputs(runtimeBridge, {
    configChain: request.configChain ?? null,
    runtimePreset: request.runtimePreset ?? null,
    runtimeConfigUrl: request.runtimeConfigUrl ?? null,
    runtimeConfig: request.runtimeConfig ?? null,
    runtimeContractPatch: buildRuntimeContractPatch(request),
  }, {
    applyRuntimePreset: runtimeBridge.applyRuntimePreset?.bind(runtimeBridge),
    applyRuntimeConfigFromUrl: runtimeBridge.applyRuntimeConfigFromUrl?.bind(runtimeBridge),
  }, options);
  assertCalibrateRuntimeCompatibility(request, runtimeBridge.getRuntimeConfig());
}

export async function runWithRuntimeIsolation(runtimeBridge, run) {
  const snapshot = snapshotRuntimeState(runtimeBridge);
  try {
    return await run();
  } finally {
    restoreRuntimeState(runtimeBridge, snapshot);
  }
}

export function buildSuiteOptions(request, surface = null) {
  const normalizedSurface = typeof surface === 'string' && surface.trim()
    ? surface.trim()
    : null;
  return {
    suite: request.suite,
    command: request.command,
    surface: normalizedSurface,
    modelId: request.modelId ?? undefined,
    trainingTests: request.trainingTests ?? undefined,
    trainingStage: request.trainingStage ?? undefined,
    trainingConfig: request.trainingConfig ?? undefined,
    stage1Artifact: request.stage1Artifact ?? undefined,
    stage1ArtifactHash: request.stage1ArtifactHash ?? undefined,
    ulArtifactDir: request.ulArtifactDir ?? undefined,
    stageAArtifact: request.stageAArtifact ?? undefined,
    stageAArtifactHash: request.stageAArtifactHash ?? undefined,
    distillArtifactDir: request.distillArtifactDir ?? undefined,
    teacherModelId: request.teacherModelId ?? undefined,
    studentModelId: request.studentModelId ?? undefined,
    distillDatasetId: request.distillDatasetId ?? undefined,
    distillDatasetPath: request.distillDatasetPath ?? undefined,
    distillLanguagePair: request.distillLanguagePair ?? undefined,
    distillSourceLangs: request.distillSourceLangs ?? undefined,
    distillTargetLangs: request.distillTargetLangs ?? undefined,
    distillPairAllowlist: request.distillPairAllowlist ?? undefined,
    strictPairContract: request.strictPairContract ?? undefined,
    distillShardIndex: request.distillShardIndex ?? undefined,
    distillShardCount: request.distillShardCount ?? undefined,
    resumeFrom: request.resumeFrom ?? undefined,
    forceResume: request.forceResume ?? undefined,
    forceResumeReason: request.forceResumeReason ?? undefined,
    forceResumeSource: request.forceResumeSource ?? undefined,
    checkpointOperator: request.checkpointOperator ?? undefined,
    trainingSchemaVersion: request.trainingSchemaVersion ?? undefined,
    trainingBenchSteps: request.trainingBenchSteps ?? undefined,
    checkpointEvery: request.checkpointEvery ?? undefined,
    workloadType: request.workloadType ?? undefined,
    modelUrl: request.modelUrl ?? undefined,
    cacheMode: request.cacheMode ?? 'warm',
    loadMode: request.loadMode ?? null,
    runtimePreset: request.runtimePreset ?? null,
    captureOutput: request.captureOutput,
    keepPipeline: request.keepPipeline,
    report: request.report || undefined,
    timestamp: request.timestamp ?? undefined,
    searchParams: request.searchParams ?? undefined,
  };
}
