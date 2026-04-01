import assert from 'node:assert/strict';
import fs from 'node:fs';

const { resolveCapabilityTransforms } = await import(
  '../../src/config/transforms/capability-transform-resolver.js'
);
const { composeTransforms, remapQ4KDecodeToGemv, remapQ4KDecodeFFNToGemv, remapQ4KDecodeAttentionToGemv, remapQ4KDecodeAttentionToFusedQ4KGemv, remapQ4KPrefillToDense } = await import(
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

const transformCtx = {
  capabilities,
  activationDtype: 'f32',
  kvDtype: 'f16',
  modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  layerTypes: conversionConfig.inference.layerPattern.layerTypes,
};

// =============================================================================
// GEMV decode is baked into the conversion config as the primary execution graph.
// Capability transforms should resolve to [] on all capable platforms.
// =============================================================================
{
  // Apple Metal
  const resolvedApple = resolveCapabilityTransforms(capabilities, {
    id: 'metal', vendor: 'apple', architecture: 'metal-3',
  }, {
    activationDtype: 'f32',
    kvDtype: 'f16',
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: conversionConfig.inference.layerPattern.layerTypes,
    hasDensePrefillProjectionKernel: true,
    hasQ4DecodeProjectionKernel: false,
    hasQ4PrefillProjectionKernel: false,
    hasAvailableQ4PrefillProjectionKernel: false,
  });
  assert.deepEqual(resolvedApple.names, [],
    'Apple Metal: no capability transforms needed (GEMV is primary path)');

  // AMD Vulkan
  const resolvedAmd = resolveCapabilityTransforms(capabilities, {
    id: 'vulkan', vendor: 'amd', architecture: 'rdna3',
  }, {
    activationDtype: 'f32',
    kvDtype: 'f16',
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: conversionConfig.inference.layerPattern.layerTypes,
    hasDensePrefillProjectionKernel: true,
    hasQ4DecodeProjectionKernel: false,
    hasQ4PrefillProjectionKernel: false,
    hasAvailableQ4PrefillProjectionKernel: false,
  });
  assert.deepEqual(resolvedAmd.names, [],
    'AMD Vulkan: no capability transforms needed (GEMV is primary path)');
}

// =============================================================================
// Direct transform functions return null on an already-GEMV execution graph
// (no fused Q4K decode kernels to remap)
// =============================================================================
{
  assert.equal(
    remapQ4KDecodeToGemv(structuredClone(execution), transformCtx),
    null,
    'remapQ4KDecodeToGemv returns null — no fused Q4K decode kernels to remap'
  );

  assert.equal(
    remapQ4KDecodeAttentionToGemv(structuredClone(execution), transformCtx),
    null,
    'remapQ4KDecodeAttentionToGemv returns null — attention projections already GEMV'
  );

  assert.equal(
    remapQ4KDecodeAttentionToFusedQ4KGemv(structuredClone(execution), transformCtx),
    null,
    'remapQ4KDecodeAttentionToFusedQ4KGemv returns null — no fused Q4K attention kernels'
  );

  assert.equal(
    remapQ4KDecodeFFNToGemv(structuredClone(execution), transformCtx),
    null,
    'remapQ4KDecodeFFNToGemv returns null — no fused Q4K FFN kernels'
  );

  assert.equal(
    remapQ4KPrefillToDense(structuredClone(execution), transformCtx),
    null,
    'remapQ4KPrefillToDense returns null — prefill already uses dense tiled kernels'
  );
}

// =============================================================================
// Verify primary graph has correct kernel assignments
// =============================================================================
{
  // Decode: all projections use GEMV subgroup
  for (const step of execution.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_gemv_subgroup.wgsl',
        `Primary graph: decode ${op} should use GEMV subgroup`
      );
    }
  }

  // Prefill: all projections use dense tiled
  for (const step of execution.prefill) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_f16w_f32a.wgsl',
        `Primary graph: prefill ${op} should use dense tiled`
      );
    }
  }

  // Prefill attention: head256
  for (const step of execution.prefill) {
    if (!Array.isArray(step)) continue;
    if (step[0] === 'attention') {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'attention_head256_f16kv.wgsl',
        'Primary graph: prefill attention should use head256'
      );
    }
  }

  // Post-layer lm_head: GEMV subgroup
  for (const step of execution.postLayer) {
    if (!Array.isArray(step)) continue;
    if (step[0] === 'lm_head') {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_gemv_subgroup.wgsl',
        'Primary graph: lm_head should use GEMV subgroup'
      );
    }
  }
}

// =============================================================================
// Verify compileExecutionV1 — zero transforms, correct kernel path
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
    runtimeCompute: { activationDtype: 'f32' },
    kernelPathPolicy: {
      mode: 'capability-aware',
      sourceScope: ['model', 'manifest', 'config'],
      allowSources: ['model', 'manifest', 'config'],
      onIncompatible: 'remap',
    },
    capabilities,
    platform: { id: 'metal', vendor: 'apple', architecture: 'metal-3' },
  });

  assert.deepEqual(compiled.appliedTransforms, [],
    'compileExecutionV1: zero transforms applied (GEMV is primary path)');

  const kp = compiled.runtimeInferencePatch.kernelPath;
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'q_proj')?.kernel,
    'matmul_gemv_subgroup.wgsl',
    'compileExecutionV1: decode q_proj uses GEMV'
  );
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'gate_proj')?.kernel,
    'matmul_gemv_subgroup.wgsl',
    'compileExecutionV1: decode gate_proj uses GEMV'
  );
  assert.equal(
    kp.prefill.steps.find((s) => s.op === 'q_proj')?.kernel,
    'matmul_f16w_f32a.wgsl',
    'compileExecutionV1: prefill q_proj uses dense tiled'
  );
  assert.equal(
    kp.prefill.steps.find((s) => s.op === 'attention')?.kernel,
    'attention_head256_f16kv.wgsl',
    'compileExecutionV1: prefill attention uses head256'
  );
}

console.log('qwen-execution-v1-gemv-diagnostic-transforms.test: ok');
