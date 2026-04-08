import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

const { createDopplerConfig } = await import('../../src/config/schema/index.js');
const { applyExecutionV1RuntimeConfig } = await import('../../src/inference/pipelines/text/execution-v1.js');
const {
  compileExecutionPlanState,
  hasFallbackExecutionPlan,
} = await import('../../src/inference/pipelines/text/execution-plan.js');

const catalog = JSON.parse(fs.readFileSync('models/catalog.json', 'utf8'));

const PROJECTION_OPS = new Set([
  'q_proj', 'k_proj', 'v_proj', 'o_proj',
  'gate_proj', 'up_proj', 'down_proj',
]);

const DENSE_PREFILL_KERNEL_FILES = new Set([
  'matmul_f16w_f32a.wgsl',
  'matmul_f16w_f32a_tiled.wgsl',
  'matmul_f16.wgsl',
  'matmul_f16_tiled.wgsl',
]);

function summarizeExecutionGraph(execution) {
  const summary = {
    hasDensePrefillProjectionKernel: false,
    hasQ4DecodeProjectionKernel: false,
    hasQ4PrefillProjectionKernel: false,
    hasAvailableQ4PrefillProjectionKernel: false,
    hasDenseProjectionKernel: false,
    hasQ4ProjectionKernel: false,
  };

  for (const step of execution?.decode ?? []) {
    if (!PROJECTION_OPS.has(step[0])) {
      continue;
    }
    const kernelEntry = execution?.kernels?.[step[1]];
    if (kernelEntry?.kernel === 'fused_matmul_q4.wgsl') {
      summary.hasQ4DecodeProjectionKernel = true;
      summary.hasQ4ProjectionKernel = true;
    }
    if (kernelEntry?.kernel?.startsWith('matmul_')) {
      summary.hasDenseProjectionKernel = true;
    }
  }

  for (const step of execution?.prefill ?? []) {
    if (!PROJECTION_OPS.has(step[0])) {
      continue;
    }
    const kernelEntry = execution?.kernels?.[step[1]];
    if (!kernelEntry) {
      continue;
    }
    if (DENSE_PREFILL_KERNEL_FILES.has(kernelEntry.kernel)) {
      summary.hasDensePrefillProjectionKernel = true;
      summary.hasDenseProjectionKernel = true;
    }
    if (kernelEntry.kernel.startsWith('fused_matmul_q4')) {
      summary.hasQ4PrefillProjectionKernel = true;
      summary.hasQ4ProjectionKernel = true;
    }
  }

  summary.hasAvailableQ4PrefillProjectionKernel = Object.values(execution?.kernels ?? {}).some(
    (entry) => entry?.kernel === 'fused_matmul_q4_batched_multicol_shared.wgsl'
      || entry?.kernel === 'fused_matmul_q4_batched.wgsl'
  );

  return summary;
}

const CLAIMED_TEXT_MODEL_IDS = catalog.models
  .filter((entry) => entry?.lifecycle?.status?.tested === 'verified')
  .filter((entry) => Array.isArray(entry?.modes) && entry.modes.some((mode) => mode === 'text' || mode === 'translate'))
  .map((entry) => entry.modelId);

const CHECKED_MODELS = [];
const REMAP_APPLIED_MODELS = [];

for (const modelId of CLAIMED_TEXT_MODEL_IDS) {
  const manifestPath = path.join('models/local', modelId, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    console.log(`claimed-text-models-failfast-contract.test: skipped ${modelId} (missing local manifest)`);
    continue;
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const graphSummary = summarizeExecutionGraph(manifest.inference?.execution);
  const runtimeConfig = createDopplerConfig({
    runtime: {
      inference: {
        kernelPathPolicy: {
          mode: 'capability-aware',
          sourceScope: ['manifest', 'model', 'config'],
          onIncompatible: 'remap',
        },
      },
    },
  }).runtime;

  const compiled = applyExecutionV1RuntimeConfig({
    runtimeConfig,
    manifest,
    modelId: manifest.modelId,
    numLayers: Number(manifest.architecture?.numLayers ?? 0),
    capabilities: {
      hasSubgroups: true,
      hasF16: true,
      hasSubgroupsF16: true,
      maxWorkgroupSize: 256,
      maxBufferSize: 1 << 30,
    },
    platform: {
      id: 'test-apple',
      vendor: 'apple',
      architecture: 'm-series',
    },
  });

  const runtimeWithDecodeLoop = {
    ...compiled.runtimeConfig,
    inference: {
      ...compiled.runtimeConfig.inference,
      session: {
        ...compiled.runtimeConfig.inference?.session,
        decodeLoop: manifest.inference?.session?.decodeLoop ?? {
          batchSize: 1,
          stopCheckMode: 'batch',
          readbackInterval: 1,
          disableCommandBatching: false,
        },
      },
    },
  };

  const planState = compileExecutionPlanState({
    runtimeConfig: runtimeWithDecodeLoop,
    resolvedKernelPath: compiled.executionV1State?.runtimeInferencePatch?.kernelPath ?? null,
    kernelPathSource: compiled.executionV1State?.runtimeInferencePatch?.kernelPathSource ?? 'none',
    fallbackKernelPath: compiled.executionV1State?.fallbackKernelPath ?? null,
    manifest,
  });

  assert.equal(
    compiled.executionV1State?.fallbackKernelPath ?? null,
    null,
    `${modelId}: claimed verified runtime must not synthesize a finiteness fallback kernel path by default`
  );
  assert.equal(
    hasFallbackExecutionPlan(planState),
    false,
    `${modelId}: claimed verified runtime must not build a fallback execution plan by default`
  );
  assert.equal(
    graphSummary.hasDenseProjectionKernel && graphSummary.hasQ4ProjectionKernel,
    false,
    `${modelId}: claimed verified base execution graphs must not pin mixed dense and fused projection steps in the same compiled manifest phase`
  );
  const remapExpected = graphSummary.hasDensePrefillProjectionKernel
    && graphSummary.hasQ4DecodeProjectionKernel
    && !graphSummary.hasQ4PrefillProjectionKernel;
  const remapApplied = compiled.executionV1State?.appliedTransforms?.includes('remapDenseQ4KPrefillToQ4Native') === true;
  assert.equal(
    remapApplied,
    remapExpected,
    `${modelId}: Q4K prefill capability remap must apply iff the manifest exposes dense prefill projections plus an explicit reusable Q4 prefill kernel`
  );
  if (remapApplied) {
    REMAP_APPLIED_MODELS.push(modelId);
  }
  CHECKED_MODELS.push(modelId);
}

assert.ok(CHECKED_MODELS.length > 0, 'No claimed verified text/translate models were available for fail-fast contract checks.');

console.log(
  `claimed-text-models-failfast-contract.test: ok (${CHECKED_MODELS.length} models checked, ` +
  `${REMAP_APPLIED_MODELS.length} Q4 prefill remaps applied)`
);
