import assert from 'node:assert/strict';
import { installNodeFileFetchShim } from '../../src/tooling/node-file-fetch.js';

installNodeFileFetchShim();

const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const { resolveKernelPath } = await import('../../src/config/kernel-path-loader.js');
const {
  compileExecutionPlanState,
  resolveActiveExecutionPlan,
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
}

{
  const enabled = isBatchDecodeEnabled({
    batchSize: 4,
    useGPU: true,
    gpuSamplingAvailable: true,
    disableMultiTokenDecode: false,
    disableCommandBatching: false,
    isBdpaPagedLayout: false,
    finitenessFallbackWindowOpen: false,
  });
  assert.equal(enabled, true);

  const disabled = isBatchDecodeEnabled({
    batchSize: 4,
    useGPU: true,
    gpuSamplingAvailable: true,
    disableMultiTokenDecode: false,
    disableCommandBatching: false,
    isBdpaPagedLayout: true,
    finitenessFallbackWindowOpen: false,
  });
  assert.equal(disabled, false);
}

{
  assert.equal(
    isDecodeRecorderEnabled({
      hasDevice: true,
      debug: false,
      disableCommandBatching: false,
      kvLayout: 'paged',
    }),
    true
  );

  assert.equal(
    isDecodeRecorderEnabled({
      hasDevice: true,
      debug: false,
      disableCommandBatching: false,
      kvLayout: 'bdpa_paged',
    }),
    false
  );
}

console.log('execution-plan.test: ok');
