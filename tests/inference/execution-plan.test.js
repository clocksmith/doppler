import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const {
  compileExecutionPlanState,
  hasFallbackExecutionPlan,
  resolveActiveExecutionPlan,
  setActiveExecutionPlan,
  activateFallbackExecutionPlan,
  resetActiveExecutionPlan,
  resolveExecutionSessionPlan,
  rebaseExecutionSessionPlan,
  isBatchDecodeEnabled,
  isDecodeRecorderEnabled,
  isProfileDecodeRecorderEnabled,
  resolveMaxBatchDecodeTokens,
} = await import('../../src/inference/pipelines/text/execution-plan.js');

const minimalKernelPath = {
  id: 'gemma3-f16-fused-f16a-online',
  name: 'gemma3-f16-fused-f16a-online',
  activationDtype: 'f16',
  decode: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
  prefill: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
};

const minimalFallbackKernelPath = {
  id: 'gemma3-f16-fused-f32a-online',
  name: 'gemma3-f16-fused-f32a-online',
  activationDtype: 'f32',
  decode: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
  prefill: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
};

const mixedPrecisionKernelPath = {
  id: 'qwen-selective-f16-primary',
  name: 'qwen-selective-f16-primary',
  activationDtype: 'f32',
  kvDtype: 'f16',
  decode: {
    steps: [
      { op: 'q_proj', kernel: 'fused_matmul_q4_multicol_f16a.wgsl', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
      { op: 'attention', kernel: 'attention_decode_online_f16.wgsl', precision: { activationDtype: 'f16', kvDtype: 'f16' } },
    ],
  },
  prefill: {
    steps: [
      { op: 'q_proj', kernel: 'fused_matmul_q4_batched_f16a.wgsl', precision: { inputDtype: 'f16', outputDtype: 'f16' } },
      { op: 'attention', kernel: 'attention_streaming_f16.wgsl', precision: { activationDtype: 'f16', kvDtype: 'f16' } },
    ],
  },
};

function createRuntimeConfig(activationDtype = 'f16', maxTokens = 256) {
  const runtimeConfig = createDopplerConfig().runtime;
  runtimeConfig.inference.compute.activationDtype = activationDtype;
  runtimeConfig.inference.compute.rangeAwareSelectiveWidening = {
    enabled: true,
    includeNonFinite: true,
    onTrigger: 'error',
    absThreshold: 65500,
  };
  runtimeConfig.inference.generation.maxTokens = maxTokens;
  runtimeConfig.inference.session.decodeLoop = {
    batchSize: 4,
    stopCheckMode: 'batch',
    readbackInterval: 1,
    readbackMode: 'sequential',
    ringTokens: 1,
    ringStop: 1,
    ringStaging: 1,
    disableCommandBatching: false,
  };
  return runtimeConfig;
}

const runtimeConfig = createRuntimeConfig('f16');
runtimeConfig.inference.compute.rangeAwareSelectiveWidening.onTrigger = 'fallback-plan';

const planState = compileExecutionPlanState({
  runtimeConfig,
  resolvedKernelPath: minimalKernelPath,
  kernelPathSource: 'model',
  fallbackKernelPath: minimalFallbackKernelPath,
});

const container = { executionPlanState: planState };

{
  assert.equal(hasFallbackExecutionPlan(container), true);
  assert.equal(hasFallbackExecutionPlan(planState), true);
}

{
  const active = resolveActiveExecutionPlan(container);
  assert.equal(active.id, 'primary');
  assert.equal(active.activationDtype, 'f16');
  assert.equal(active.kernelPathId, 'gemma3-f16-fused-f16a-online');
  assert.equal(active.finitenessGuardEnabled, true);
  assert.equal(active.finitenessOnTrigger, 'fallback-plan');
  assert.equal(active.readbackMode, 'sequential');
}

{
  const fallback = activateFallbackExecutionPlan(container);
  assert.ok(fallback);
  assert.equal(fallback.id, 'finiteness_fallback');
  assert.equal(fallback.activationDtype, 'f32');
  assert.equal(fallback.kernelPathId, 'gemma3-f16-fused-f32a-online');
  const active = resolveActiveExecutionPlan(container);
  assert.equal(active.id, 'finiteness_fallback');
  const activeFromPlanState = resolveActiveExecutionPlan(planState);
  assert.equal(activeFromPlanState.id, 'finiteness_fallback');
}

{
  const primarySession = resolveExecutionSessionPlan(container, {
    batchSize: 6,
    disableCommandBatching: true,
    readbackInterval: 3,
    ringTokens: 2,
    ringStop: 2,
    ringStaging: 2,
  });
  assert.equal(primarySession.activationDtype, 'f32');
  assert.equal(primarySession.batchSize, 6);
  assert.equal(primarySession.disableCommandBatching, true);
  assert.equal(primarySession.readbackInterval, 3);
  assert.equal(primarySession.ringTokens, 2);
  assert.equal(primarySession.ringStop, 2);
  assert.equal(primarySession.ringStaging, 2);
  assert.equal(primarySession.overrides.readbackInterval, 3);
  assert.equal(primarySession.overrides.ringTokens, 2);
  assert.equal(primarySession.overrides.ringStop, 2);
  assert.equal(primarySession.overrides.ringStaging, 2);

  resetActiveExecutionPlan(container);
  const rebasedPrimary = rebaseExecutionSessionPlan(container, primarySession);
  assert.equal(rebasedPrimary.activationDtype, 'f16');
  assert.equal(rebasedPrimary.batchSize, 6);
  assert.equal(rebasedPrimary.disableCommandBatching, true);

  const rebasedDefaults = rebaseExecutionSessionPlan(container, null);
  assert.equal(rebasedDefaults.planId, 'primary');
  assert.equal(rebasedDefaults.batchSize, planState.primaryPlan.defaultBatchSize);
  assert.equal(
    rebasedDefaults.disableCommandBatching,
    planState.primaryPlan.defaultDisableCommandBatching
  );
}

{
  const enabledConfig = {
    batchSize: 4,
    useGPU: true,
    gpuSamplingAvailable: true,
    disableMultiTokenDecode: false,
    disableCommandBatching: false,
    isBdpaPagedLayout: false,
    finitenessFallbackWindowOpen: false,
    hasLinearAttentionLayers: false,
    selfSpeculationEnabled: false,
    hasRangeBackedPerLayerInputs: false,
  };
  const enabled = isBatchDecodeEnabled(enabledConfig);
  assert.equal(enabled, true);

  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, batchSize: 1 }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, useGPU: false }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, gpuSamplingAvailable: false }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, disableMultiTokenDecode: true }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, disableMultiTokenDecode: undefined }), true);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, disableCommandBatching: true }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, isBdpaPagedLayout: true }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, finitenessFallbackWindowOpen: true }), false);
  assert.equal(
    isBatchDecodeEnabled({
      ...enabledConfig,
      hasLinearAttentionLayers: true,
      selfSpeculationEnabled: false,
      hasRangeBackedPerLayerInputs: false,
    }),
    true
  );
  assert.equal(
    isBatchDecodeEnabled({
      ...enabledConfig,
      hasLinearAttentionLayers: true,
      selfSpeculationEnabled: true,
      hasRangeBackedPerLayerInputs: false,
    }),
    true
  );
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, hasRangeBackedPerLayerInputs: true }), true);
  assert.equal(
    isBatchDecodeEnabled({
      ...enabledConfig,
      hasRangeBackedPerLayerInputs: true,
      selfSpeculationEnabled: true,
    }),
    false
  );
  assert.equal(resolveMaxBatchDecodeTokens({ hasHotVocabularyBatchDecode: true }), 1);
  assert.equal(resolveMaxBatchDecodeTokens({ hasLinearAttentionLayers: true }), 32);
  assert.equal(resolveMaxBatchDecodeTokens({ hasGpuSplitPerLayerInputs: false }), null);
  assert.equal(resolveMaxBatchDecodeTokens({ hasGpuSplitPerLayerInputs: true }), 4);
}

{
  const runtimeConfigSessionWins = createRuntimeConfig('f16');
  runtimeConfigSessionWins.inference.session.decodeLoop.disableCommandBatching = true;

  const sessionWinsPlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigSessionWins,
    resolvedKernelPath: minimalKernelPath,
    kernelPathSource: 'model',
    fallbackKernelPath: minimalFallbackKernelPath,
  });

  assert.equal(sessionWinsPlanState.primaryPlan.defaultDisableCommandBatching, true);
  assert.equal(
    resolveExecutionSessionPlan(sessionWinsPlanState).disableCommandBatching,
    true
  );
}

{
  const runtimeConfigLegacyGenerationField = createRuntimeConfig('f16');
  runtimeConfigLegacyGenerationField.inference.generation.disableCommandBatching = true;

  assert.throws(
    () => compileExecutionPlanState({
      runtimeConfig: runtimeConfigLegacyGenerationField,
      resolvedKernelPath: minimalKernelPath,
      kernelPathSource: 'model',
      fallbackKernelPath: minimalFallbackKernelPath,
    }),
    /runtime\.inference\.generation\.disableCommandBatching is removed/
  );
}

{
  const enabledConfig = {
    hasDevice: true,
    debug: false,
    disableCommandBatching: false,
    kvLayout: 'paged',
  };

  assert.equal(isDecodeRecorderEnabled(enabledConfig), true);
  assert.equal(isDecodeRecorderEnabled({ ...enabledConfig, kvLayout: 'bdpa_paged' }), false);
  assert.equal(isDecodeRecorderEnabled({ ...enabledConfig, debug: true }), false);
  assert.equal(isDecodeRecorderEnabled({ ...enabledConfig, disableCommandBatching: true }), false);
  assert.equal(isDecodeRecorderEnabled({ ...enabledConfig, hasDevice: false }), false);
  assert.equal(isProfileDecodeRecorderEnabled(enabledConfig), true);
  assert.equal(isProfileDecodeRecorderEnabled({ ...enabledConfig, kvLayout: 'bdpa_paged' }), false);
  assert.equal(isProfileDecodeRecorderEnabled({ ...enabledConfig, debug: true }), false);
  assert.equal(isProfileDecodeRecorderEnabled({ ...enabledConfig, disableCommandBatching: true }), true);
  assert.equal(isProfileDecodeRecorderEnabled({ ...enabledConfig, hasDevice: false }), false);
}

{
  assert.throws(
    () => setActiveExecutionPlan(container, 'not-a-plan'),
    /unknown plan id "not-a-plan"/
  );
}

{
  const runtimeConfigFailFast = createRuntimeConfig('f16');
  const failFastPlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigFailFast,
    resolvedKernelPath: minimalKernelPath,
    kernelPathSource: 'model',
  });

  assert.equal(hasFallbackExecutionPlan(failFastPlanState), false);
  assert.equal(failFastPlanState.primaryPlan.finitenessOnTrigger, 'error');
  assert.equal(activateFallbackExecutionPlan(failFastPlanState), null);
}

{
  const runtimeConfigNoFallback = createRuntimeConfig('f32');
  const noFallbackPlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigNoFallback,
    resolvedKernelPath: {
      id: 'gemma3-f16-fused-f32a-online',
      name: 'gemma3-f16-fused-f32a-online',
      activationDtype: 'f32',
      decode: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
      prefill: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
    },
    kernelPathSource: 'model',
  });

  assert.equal(hasFallbackExecutionPlan(noFallbackPlanState), false);
  assert.equal(activateFallbackExecutionPlan(noFallbackPlanState), null);
  assert.equal(resolveActiveExecutionPlan(noFallbackPlanState).id, 'primary');
}

{
  const runtimeConfigSelectiveF16 = createRuntimeConfig('f32');
  runtimeConfigSelectiveF16.inference.compute.rangeAwareSelectiveWidening.onTrigger = 'fallback-plan';
  const selectivePlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigSelectiveF16,
    resolvedKernelPath: mixedPrecisionKernelPath,
    fallbackKernelPath: minimalFallbackKernelPath,
    kernelPathSource: 'execution-v1',
  });

  assert.equal(selectivePlanState.primaryPlan.activationDtype, 'f32');
  assert.equal(selectivePlanState.primaryPlan.finitenessGuardEnabled, false);
  assert.equal(hasFallbackExecutionPlan(selectivePlanState), true);
  assert.equal(selectivePlanState.fallbackPlan?.kernelPathId, minimalFallbackKernelPath.id);
}

{
  const runtimeConfigMissingRule = createRuntimeConfig('f16');
  runtimeConfigMissingRule.inference.compute.rangeAwareSelectiveWidening.onTrigger = 'fallback-plan';
  assert.throws(
    () => compileExecutionPlanState({
      runtimeConfig: runtimeConfigMissingRule,
      resolvedKernelPath: {
        id: 'missing-fallback-path',
        activationDtype: 'f16',
        decode: { steps: [] },
      },
      kernelPathSource: 'model',
    }),
    /finiteness fallback kernel path required for "missing-fallback-path"/
  );
}

{
  const runtimeConfigMissingId = createRuntimeConfig('f16');
  runtimeConfigMissingId.inference.compute.rangeAwareSelectiveWidening.onTrigger = 'fallback-plan';
  const noKernelPathPlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigMissingId,
    resolvedKernelPath: {
      activationDtype: 'f16',
      decode: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
      prefill: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
    },
    fallbackKernelPath: minimalFallbackKernelPath,
    kernelPathSource: 'model',
  });

  assert.equal(noKernelPathPlanState.primaryPlan.kernelPathId, null);
  assert.equal(hasFallbackExecutionPlan(noKernelPathPlanState), true);
  const fallback = activateFallbackExecutionPlan(noKernelPathPlanState);
  assert.ok(fallback);
  assert.equal(fallback.kernelPathId, minimalFallbackKernelPath.id);
  assert.equal(fallback.kernelPathSource, 'execution-v1-transform');
  assert.equal(fallback.activationDtype, 'f32');
}

{
  const runtimeConfigInlineFallback = createRuntimeConfig('f16');
  runtimeConfigInlineFallback.inference.compute.rangeAwareSelectiveWidening.onTrigger = 'fallback-plan';
  const inlinePlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigInlineFallback,
    resolvedKernelPath: {
      id: 'gemma-inline-fallback',
      activationDtype: 'f16',
      finitenessFallbackKernelPathId: 'gemma3-q4k-dequant-f32a-nosubgroups',
      decode: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
      prefill: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
    },
    kernelPathSource: 'config',
    fallbackKernelPath: {
      id: 'gemma-inline-fallback-transform',
      activationDtype: 'f32',
      decode: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
      prefill: { steps: [{ op: 'noop', kernel: 'noop.wgsl' }] },
    },
  });

  const fallback = activateFallbackExecutionPlan(inlinePlanState);
  assert.ok(fallback);
  assert.equal(fallback.kernelPathId, 'gemma-inline-fallback-transform');
  assert.equal(fallback.kernelPathSource, 'execution-v1-transform');
  assert.equal(fallback.activationDtype, 'f32');
}

{
  assert.throws(
    () => resolveExecutionSessionPlan(container, { batchSize: 0 }),
    /batchSize must be a positive integer/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { batchSize: 2.5 }),
    /batchSize must be a positive integer/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { maxTokens: '16' }),
    /maxTokens must be a positive integer/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { stopCheckMode: 'frame' }),
    /stopCheckMode must be "batch" or "per-token"/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { disableCommandBatching: 'true' }),
    /disableCommandBatching must be boolean/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { disableMultiTokenDecode: 1 }),
    /disableMultiTokenDecode must be boolean/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { readbackInterval: 0 }),
    /readbackInterval must be a positive integer/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { ringTokens: 0 }),
    /ringTokens must be a positive integer/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { ringStop: 0 }),
    /ringStop must be a positive integer/
  );
  assert.throws(
    () => resolveExecutionSessionPlan(container, { ringStaging: 0 }),
    /ringStaging must be a positive integer/
  );
}

{
  const runtimeConfigInvalidReadbackMode = createRuntimeConfig('f16');
  runtimeConfigInvalidReadbackMode.inference.session.decodeLoop.readbackMode = 'invalid';

  assert.throws(
    () => compileExecutionPlanState({
      runtimeConfig: runtimeConfigInvalidReadbackMode,
      resolvedKernelPath: minimalKernelPath,
      kernelPathSource: 'model',
      fallbackKernelPath: minimalFallbackKernelPath,
    }),
    /readbackMode must be one of sequential, overlapped, auto/
  );
}

console.log('execution-plan.test: ok');
