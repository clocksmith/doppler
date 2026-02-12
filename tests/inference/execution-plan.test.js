import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const { resolveKernelPath } = await import('../../src/config/kernel-path-loader.js');
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
} = await import('../../src/inference/pipelines/text/execution-plan.js');

const runtimeConfig = createDopplerConfig().runtime;
runtimeConfig.inference.compute.activationDtype = 'f16';

const planState = compileExecutionPlanState({
  runtimeConfig,
  resolvedKernelPath: resolveKernelPath('gemma3-f16-f16a'),
  kernelPathSource: 'model',
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
  assert.equal(active.kernelPathId, 'gemma3-f16-f16a');
  assert.equal(active.finitenessGuardEnabled, true);
}

{
  const fallback = activateFallbackExecutionPlan(container);
  assert.ok(fallback);
  assert.equal(fallback.id, 'finiteness_fallback');
  assert.equal(fallback.activationDtype, 'f32');
  assert.equal(fallback.kernelPathId, 'gemma3-f16-f32a');
  const active = resolveActiveExecutionPlan(container);
  assert.equal(active.id, 'finiteness_fallback');
  const activeFromPlanState = resolveActiveExecutionPlan(planState);
  assert.equal(activeFromPlanState.id, 'finiteness_fallback');
}

{
  const primarySession = resolveExecutionSessionPlan(container, {
    batchSize: 6,
    disableCommandBatching: true,
  });
  assert.equal(primarySession.activationDtype, 'f32');
  assert.equal(primarySession.batchSize, 6);
  assert.equal(primarySession.disableCommandBatching, true);

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
  };
  const enabled = isBatchDecodeEnabled(enabledConfig);
  assert.equal(enabled, true);

  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, batchSize: 1 }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, useGPU: false }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, gpuSamplingAvailable: false }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, disableMultiTokenDecode: true }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, disableCommandBatching: true }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, isBdpaPagedLayout: true }), false);
  assert.equal(isBatchDecodeEnabled({ ...enabledConfig, finitenessFallbackWindowOpen: true }), false);
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
}

{
  assert.throws(
    () => setActiveExecutionPlan(container, 'not-a-plan'),
    /unknown plan id "not-a-plan"/
  );
}

{
  const runtimeConfigNoFallback = createDopplerConfig().runtime;
  runtimeConfigNoFallback.inference.compute.activationDtype = 'f32';
  const noFallbackPlanState = compileExecutionPlanState({
    runtimeConfig: runtimeConfigNoFallback,
    resolvedKernelPath: resolveKernelPath('gemma3-f16-f32a'),
    kernelPathSource: 'model',
  });

  assert.equal(hasFallbackExecutionPlan(noFallbackPlanState), false);
  assert.equal(activateFallbackExecutionPlan(noFallbackPlanState), null);
  assert.equal(resolveActiveExecutionPlan(noFallbackPlanState).id, 'primary');
}

console.log('execution-plan.test: ok');
