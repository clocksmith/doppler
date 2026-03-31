import assert from 'node:assert/strict';
import fs from 'node:fs';

const { resolveCapabilityTransforms } = await import(
  '../../src/config/transforms/capability-transform-resolver.js'
);
const { composeTransforms, remapQ4KDecodeFFNToGemv, remapQ4KDecodeAttentionToGemv, remapQ4KDecodeAttentionToFusedQ4KGemv } = await import(
  '../../src/config/transforms/execution-graph-transforms.js'
);

const conversionConfig = JSON.parse(
  fs.readFileSync(
    new URL('../../src/config/conversion/qwen3/qwen-3-5-0-8b-q4k-ehaf16.json', import.meta.url),
    'utf8'
  )
);

const execution = conversionConfig.execution;
const capabilities = {
  hasSubgroups: true,
  hasF16: true,
  maxWorkgroupStorageSize: 32768,
};
const platform = {
  id: 'metal',
  vendor: 'apple',
  architecture: 'metal-3',
};
const baseGraphContext = {
  activationDtype: 'f32',
  kvDtype: 'f16',
  modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  layerTypes: conversionConfig.inference.layerPattern.layerTypes,
  hasDensePrefillProjectionKernel: false,
  hasQ4DecodeProjectionKernel: true,
  hasQ4PrefillProjectionKernel: true,
  hasAvailableQ4PrefillProjectionKernel: true,
};

const transformCtx = {
  capabilities,
  platform,
  activationDtype: 'f32',
  kvDtype: 'f16',
  modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  layerTypes: conversionConfig.inference.layerPattern.layerTypes,
};

// =============================================================================
// Default chain: full GEMV decode for all projections
// =============================================================================
{
  const resolved = resolveCapabilityTransforms(capabilities, platform, baseGraphContext);
  assert.deepEqual(
    resolved.names,
    ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeToGemv'],
    'Default chain should use f16 GEMV for all decode projections'
  );

  const composed = composeTransforms(...resolved.transforms);
  const graph = composed(execution, transformCtx);
  assert.ok(graph, 'Full GEMV chain should produce a transformed graph');

  // All decode projections: GEMV
  for (const step of graph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = graph.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_gemv_subgroup.wgsl',
        `Default: decode ${op} should use GEMV`
      );
    }
  }

  // Prefill projections should be dense (from remapQ4KPrefillToDense)
  for (const step of graph.prefill) {
    if (!Array.isArray(step)) continue;
    if (step[0] === 'q_proj') {
      const entry = graph.kernels[step[1]];
      assert.ok(
        entry.kernel.startsWith('matmul_f16w_f32a'),
        `Default: prefill q_proj should use dense matmul, got ${entry.kernel}`
      );
    }
  }
}

// =============================================================================
// Direct transform function tests (not rule-resolved)
// =============================================================================
// remapQ4KDecodeAttentionToGemv, remapQ4KDecodeAttentionToFusedQ4KGemv, and
// remapQ4KDecodeFFNToGemv are tested directly.
{
  // Attention-only f16 GEMV (diagnostic — causes precision loss)
  const attnGraph = remapQ4KDecodeAttentionToGemv(structuredClone(execution), transformCtx);
  assert.ok(attnGraph, 'remapQ4KDecodeAttentionToGemv should produce a transformed graph');

  for (const step of attnGraph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj'].includes(op)) {
      const entry = attnGraph.kernels[step[1]];
      assert.equal(entry.kernel, 'matmul_gemv_subgroup.wgsl',
        `remapQ4KDecodeAttentionToGemv: decode ${op} should use GEMV`);
    }
    if (['gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = attnGraph.kernels[step[1]];
      assert.ok(entry.kernel.startsWith('fused_matmul_q4'),
        `remapQ4KDecodeAttentionToGemv: decode ${op} should remain fused Q4K`);
    }
  }

  // Attention-only fused Q4K GEMV (production — no precision loss)
  const attnFusedGraph = remapQ4KDecodeAttentionToFusedQ4KGemv(structuredClone(execution), transformCtx);
  assert.ok(attnFusedGraph, 'remapQ4KDecodeAttentionToFusedQ4KGemv should produce a transformed graph');

  for (const step of attnFusedGraph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj'].includes(op)) {
      const entry = attnFusedGraph.kernels[step[1]];
      assert.equal(entry.kernel, 'fused_matmul_q4.wgsl',
        `remapQ4KDecodeAttentionToFusedQ4KGemv: decode ${op} should use fused Q4K`);
      assert.equal(entry.entry, 'main_gemv',
        `remapQ4KDecodeAttentionToFusedQ4KGemv: decode ${op} should use main_gemv entry`);
    }
    if (['gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = attnFusedGraph.kernels[step[1]];
      assert.ok(entry.kernel.startsWith('fused_matmul_q4'),
        `remapQ4KDecodeAttentionToFusedQ4KGemv: decode ${op} should remain original fused Q4K`);
      assert.notEqual(entry.entry, 'main_gemv',
        `remapQ4KDecodeAttentionToFusedQ4KGemv: decode ${op} should NOT use main_gemv`);
    }
  }

  // FFN-only GEMV
  const ffnGraph = remapQ4KDecodeFFNToGemv(structuredClone(execution), transformCtx);
  assert.ok(ffnGraph, 'remapQ4KDecodeFFNToGemv should produce a transformed graph');

  for (const step of ffnGraph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj'].includes(op)) {
      const entry = ffnGraph.kernels[step[1]];
      assert.ok(entry.kernel.startsWith('fused_matmul_q4'),
        `remapQ4KDecodeFFNToGemv: decode ${op} should remain fused Q4K`);
    }
    if (['gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = ffnGraph.kernels[step[1]];
      assert.equal(entry.kernel, 'matmul_gemv_subgroup.wgsl',
        `remapQ4KDecodeFFNToGemv: decode ${op} should use GEMV`);
    }
  }
}

// =============================================================================
// Verify compileExecutionV1 default path — full f16 GEMV decode
// =============================================================================
{
  const { compileExecutionV1 } = await import(
    '../../src/inference/pipelines/text/execution-v1.js'
  );

  const manifestInference = {
    schema: 'doppler.execution/v1',
    session: conversionConfig.session,
    layerPattern: conversionConfig.inference.layerPattern,
    execution: conversionConfig.execution,
  };

  const compiled = compileExecutionV1({
    manifestInference,
    modelId: conversionConfig.output.modelBaseId,
    numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
    runtimeSession: conversionConfig.session,
    runtimeCompute: {
      activationDtype: 'f32',
    },
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      allowSources: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    },
    capabilities,
    platform,
  });
  assert.deepEqual(
    compiled.appliedTransforms,
    ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeToGemv'],
    'compileExecutionV1 default should select full f16 GEMV decode chain'
  );

  const kp = compiled.runtimeInferencePatch.kernelPath;
  // All decode projections: GEMV
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'q_proj')?.kernel,
    'matmul_gemv_subgroup.wgsl',
    'compileExecutionV1 default: decode q_proj should use GEMV'
  );
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'gate_proj')?.kernel,
    'matmul_gemv_subgroup.wgsl',
    'compileExecutionV1 default: decode gate_proj should use GEMV'
  );
}

console.log('qwen-execution-v1-gemv-diagnostic-transforms.test: ok');
