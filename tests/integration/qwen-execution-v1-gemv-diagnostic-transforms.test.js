import assert from 'node:assert/strict';
import fs from 'node:fs';

const { resolveCapabilityTransforms } = await import(
  '../../src/config/transforms/capability-transform-resolver.js'
);
const { composeTransforms } = await import(
  '../../src/config/transforms/execution-graph-transforms.js'
);
const { expandExecutionV1 } = await import(
  '../../src/config/schema/execution-v1.schema.js'
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
// Full GEMV chain (default — no diagnosticGemvScope)
// =============================================================================
{
  const resolved = resolveCapabilityTransforms(capabilities, platform, baseGraphContext);
  assert.deepEqual(
    resolved.names,
    ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeToGemv'],
    'Default chain should be full GEMV'
  );

  const composed = composeTransforms(...resolved.transforms);
  const graph = composed(execution, transformCtx);
  assert.ok(graph, 'Full GEMV chain should produce a transformed graph');

  // All decode projections should reference GEMV kernel
  for (const step of graph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = graph.kernels[step[1]];
      assert.equal(entry.kernel, 'matmul_gemv_subgroup.wgsl', `Full GEMV: decode ${op} should use GEMV`);
    }
  }

  // Prefill projections should be dense (from remapQ4KPrefillToDense)
  for (const step of graph.prefill) {
    if (!Array.isArray(step)) continue;
    if (step[0] === 'q_proj') {
      const entry = graph.kernels[step[1]];
      assert.ok(
        entry.kernel.startsWith('matmul_f16w_f32a'),
        `Full GEMV: prefill q_proj should use dense matmul, got ${entry.kernel}`
      );
    }
  }
}

// =============================================================================
// Attention-only GEMV diagnostic
// =============================================================================
{
  const resolved = resolveCapabilityTransforms(capabilities, platform, {
    ...baseGraphContext,
    diagnosticGemvScope: 'attention',
  });
  assert.deepEqual(
    resolved.names,
    ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeAttentionToGemv'],
    'Attention diagnostic chain should use remapQ4KDecodeAttentionToGemv'
  );

  const composed = composeTransforms(...resolved.transforms);
  const graph = composed(execution, transformCtx);
  assert.ok(graph, 'Attention GEMV diagnostic chain should produce a transformed graph');

  // Attention projections: GEMV
  for (const step of graph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj'].includes(op)) {
      const entry = graph.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_gemv_subgroup.wgsl',
        `Attn diagnostic: decode ${op} should use GEMV`
      );
    }
  }

  // FFN projections: still fused Q4K
  for (const step of graph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = graph.kernels[step[1]];
      assert.ok(
        entry.kernel.startsWith('fused_matmul_q4'),
        `Attn diagnostic: decode ${op} should remain fused Q4K, got ${entry.kernel}`
      );
    }
  }
}

// =============================================================================
// FFN-only GEMV diagnostic
// =============================================================================
{
  const resolved = resolveCapabilityTransforms(capabilities, platform, {
    ...baseGraphContext,
    diagnosticGemvScope: 'ffn',
  });
  assert.deepEqual(
    resolved.names,
    ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeFFNToGemv'],
    'FFN diagnostic chain should use remapQ4KDecodeFFNToGemv'
  );

  const composed = composeTransforms(...resolved.transforms);
  const graph = composed(execution, transformCtx);
  assert.ok(graph, 'FFN GEMV diagnostic chain should produce a transformed graph');

  // Attention projections: still fused Q4K
  for (const step of graph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['q_proj', 'k_proj', 'v_proj', 'o_proj'].includes(op)) {
      const entry = graph.kernels[step[1]];
      assert.ok(
        entry.kernel.startsWith('fused_matmul_q4'),
        `FFN diagnostic: decode ${op} should remain fused Q4K, got ${entry.kernel}`
      );
    }
  }

  // FFN projections: GEMV
  for (const step of graph.decode) {
    if (!Array.isArray(step)) continue;
    const op = step[0];
    if (['gate_proj', 'up_proj', 'down_proj'].includes(op)) {
      const entry = graph.kernels[step[1]];
      assert.equal(
        entry.kernel, 'matmul_gemv_subgroup.wgsl',
        `FFN diagnostic: decode ${op} should use GEMV`
      );
    }
  }
}

// =============================================================================
// Verify diagnostic chains preserve mixed materialization
// =============================================================================
// When either attention or FFN keeps fused Q4K, the kernel path should still
// contain fused_matmul_q4 references, keeping isKernelPathFusedQ4K = true
// and therefore mixed materialization mode.
{
  for (const scope of ['attention', 'ffn']) {
    const resolved = resolveCapabilityTransforms(capabilities, platform, {
      ...baseGraphContext,
      diagnosticGemvScope: scope,
    });
    const composed = composeTransforms(...resolved.transforms);
    const graph = composed(execution, transformCtx);

    let hasFusedQ4K = false;
    for (const step of [...graph.decode, ...graph.prefill]) {
      if (!Array.isArray(step)) continue;
      const entry = graph.kernels[step[1]];
      if (entry && entry.kernel.startsWith('fused_matmul_q4')) {
        hasFusedQ4K = true;
        break;
      }
    }
    assert.ok(
      hasFusedQ4K,
      `Diagnostic scope="${scope}" should preserve fused Q4K references for mixed materialization`
    );
  }
}

// =============================================================================
// Verify compileExecutionV1 pass-through for diagnosticGemvScope
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

  // Attention-only via compileExecutionV1
  const attnCompiled = compileExecutionV1({
    manifestInference,
    modelId: conversionConfig.output.modelBaseId,
    numLayers: conversionConfig.inference.layerPattern.layerTypes.length,
    runtimeSession: conversionConfig.session,
    runtimeCompute: {
      activationDtype: 'f32',
      diagnosticGemvScope: 'attention',
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
    attnCompiled.appliedTransforms,
    ['useHead256PrefillAttention', 'remapQ4KPrefillToDense', 'remapQ4KDecodeAttentionToGemv'],
    'compileExecutionV1 with diagnosticGemvScope="attention" should select attention-only chain'
  );

  const kp = attnCompiled.runtimeInferencePatch.kernelPath;
  // Attention projections: GEMV
  assert.equal(
    kp.decode.steps.find((s) => s.op === 'q_proj')?.kernel,
    'matmul_gemv_subgroup.wgsl',
    'compileExecutionV1 attn diagnostic: decode q_proj should use GEMV'
  );
  // FFN projections: still fused Q4K
  assert.ok(
    kp.decode.steps.find((s) => s.op === 'gate_proj')?.kernel.startsWith('fused_matmul_q4'),
    'compileExecutionV1 attn diagnostic: decode gate_proj should remain fused Q4K'
  );
}

console.log('qwen-execution-v1-gemv-diagnostic-transforms.test: ok');
