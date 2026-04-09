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
  activationDtype: 'f16',
  kvDtype: 'f16',
  modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
  layerTypes: conversionConfig.inference.layerPattern.layerTypes,
};

const diagnosticTransformCtx = {
  ...transformCtx,
  activationDtype: 'f32',
};

// =============================================================================
// Fused Q4 decode/prefill is baked into the conversion config as the primary
// execution graph.
// Capability transforms should resolve to [] on all capable platforms.
// =============================================================================
{
  // Apple Metal
  const resolvedApple = resolveCapabilityTransforms(capabilities, {
    id: 'metal', vendor: 'apple', architecture: 'metal-3',
  }, {
    activationDtype: 'f16',
    kvDtype: 'f16',
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: conversionConfig.inference.layerPattern.layerTypes,
    hasDensePrefillProjectionKernel: true,
    hasQ4DecodeProjectionKernel: true,
    hasQ4PrefillProjectionKernel: true,
    hasAvailableQ4PrefillProjectionKernel: true,
    requiresF16ActivationNarrowing: false,
  });
  assert.deepEqual(resolvedApple.names, [],
    'Apple Metal: no capability transforms needed (fused Q4 is primary path)');

  // AMD Vulkan
  const resolvedAmd = resolveCapabilityTransforms(capabilities, {
    id: 'vulkan', vendor: 'amd', architecture: 'rdna3',
  }, {
    activationDtype: 'f16',
    kvDtype: 'f16',
    modelId: 'qwen-3-5-0-8b-q4k-ehaf16',
    layerTypes: conversionConfig.inference.layerPattern.layerTypes,
    hasDensePrefillProjectionKernel: true,
    hasQ4DecodeProjectionKernel: true,
    hasQ4PrefillProjectionKernel: true,
    hasAvailableQ4PrefillProjectionKernel: true,
    requiresF16ActivationNarrowing: false,
  });
  assert.deepEqual(resolvedAmd.names, [],
    'AMD Vulkan: no capability transforms needed (fused Q4 is primary path)');
}

// =============================================================================
// Direct transform functions remain available as explicit diagnostic transforms
// on the fused-Q4 primary graph.
// =============================================================================
{
  assert.ok(
    remapQ4KDecodeToGemv(structuredClone(execution), diagnosticTransformCtx),
    'remapQ4KDecodeToGemv should produce an explicit GEMV diagnostic graph'
  );
  assert.ok(
    remapQ4KDecodeAttentionToGemv(structuredClone(execution), diagnosticTransformCtx),
    'remapQ4KDecodeAttentionToGemv should produce an attention-only GEMV diagnostic graph'
  );
  assert.ok(
    remapQ4KDecodeAttentionToFusedQ4KGemv(structuredClone(execution), diagnosticTransformCtx),
    'remapQ4KDecodeAttentionToFusedQ4KGemv should produce a fused-Q4K-GEMV diagnostic graph'
  );
  assert.ok(
    remapQ4KDecodeFFNToGemv(structuredClone(execution), diagnosticTransformCtx),
    'remapQ4KDecodeFFNToGemv should produce an FFN-only GEMV diagnostic graph'
  );
  assert.ok(
    remapQ4KPrefillToDense(structuredClone(execution), diagnosticTransformCtx),
    'remapQ4KPrefillToDense should produce a dense-prefill diagnostic graph'
  );
}

// =============================================================================
// Verify primary graph has correct kernel assignments
// =============================================================================
{
  // Decode: all projections use fused Q4 multicol f16a
  for (const step of execution.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'fused_matmul_q4_multicol_f16a.wgsl',
        `Primary graph: decode ${op} should use fused_matmul_q4_multicol_f16a.wgsl`
      );
    }
  }

  // Prefill: all projections use fused batched Q4 f16a
  for (const step of execution.prefill) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'fused_matmul_q4_batched_f16a.wgsl',
        `Primary graph: prefill ${op} should use fused_matmul_q4_batched_f16a.wgsl`
      );
    }
  }

  // Prefill attention: small-tiled
  for (const step of execution.prefill) {
    if (!Array.isArray(step)) continue;
    if (step[0] === 'attention') {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'attention_small_f16.wgsl',
        'Primary graph: prefill attention should use small-tiled attention'
      );
    }
  }

  // Post-layer lm_head: GEMV subgroup
  for (const step of execution.postLayer) {
    if (!Array.isArray(step)) continue;
    if (step[0] === 'lm_head') {
      const entry = execution.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_gemv_subgroup_f16a.wgsl',
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
    runtimeCompute: { activationDtype: 'f16' },
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
    'compileExecutionV1: zero transforms applied (fused Q4 is primary path)');

  const kp = compiled.runtimeInferencePatch.kernelPath;
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'q_proj')?.kernel,
    'fused_matmul_q4_multicol_f16a.wgsl',
    'compileExecutionV1: decode q_proj uses fused Q4'
  );
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'gate_proj')?.kernel,
    'fused_matmul_q4_multicol_f16a.wgsl',
    'compileExecutionV1: decode gate_proj uses fused Q4'
  );
  assert.equal(
    kp.prefill.steps.find((s) => s.op === 'q_proj')?.kernel,
    'fused_matmul_q4_batched_f16a.wgsl',
    'compileExecutionV1: prefill q_proj uses fused batched Q4'
  );
  assert.equal(
    kp.prefill.steps.find((s) => s.op === 'attention')?.kernel,
    'attention_small_f16.wgsl',
    'compileExecutionV1: prefill attention uses small-tiled attention'
  );
}

console.log('qwen-execution-v1-gemv-diagnostic-transforms.test: ok');
