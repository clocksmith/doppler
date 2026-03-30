import assert from 'node:assert/strict';

import {
  formatKernelPath,
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  getKernelPathActivationDtype,
  getKernelPathAttentionVariant,
  getKernelPathKVDtype,
  getKernelPathMatmulConstants,
  getKernelPathMatmulPrecision,
  getKernelPathMatmulVariant,
  getKernelPathOutputDtype,
  getKernelPathStats,
  getKernelPathStrict,
  getLayerSteps,
  isActiveKernelPathDequant,
  isActiveKernelPathFusedQ4K,
  isKernelPathDequant,
  isKernelPathFusedQ4K,
  kernelPathRequiresF32MatmulWeights,
  resolveKernelPath,
  resolveWeightRef,
  setActiveKernelPath,
  validateKernelPath,
} from '../../src/config/kernel-path-loader.js';

const originalPath = getActiveKernelPath();
const originalSource = getActiveKernelPathSource();
const originalPolicy = getActiveKernelPathPolicy();

try {
  // String kernel path IDs are no longer supported (registry removed)
  assert.throws(
    () => resolveKernelPath('missing-kernel-path-id'),
    /no longer supported/
  );
  assert.equal(resolveKernelPath(null), null);

  // Inline kernel path objects for testing
  const fusedPath = {
    id: 'gemma2-q4k-fused-f32a',
    name: 'Gemma2 Q4K Fused F32A',
    activationDtype: 'f32',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'fused_matmul_q4k.wgsl' },
        { op: 'attention', kernel: 'attention_decode_f16.wgsl', entry: 'attention_decode' },
        { op: 'ffn_gate_up', kernel: 'fused_matmul_q4k.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_dequant_f32a.wgsl' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'fused_matmul_q4k.wgsl' },
        { op: 'attention', kernel: 'attention_prefill_f16.wgsl' },
        { op: 'ffn_gate_up', kernel: 'fused_matmul_q4k.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_dequant_f32a.wgsl' },
      ],
    },
    postLayer: [
      { op: 'lm_head', kernel: 'matmul_f16.wgsl' },
    ],
  };

  const dequantPath = {
    id: 'gemma2-q4k-dequant-f32a-nosubgroups',
    name: 'Gemma2 Q4K Dequant F32A',
    activationDtype: 'f32',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_dequant_f32a.wgsl' },
        { op: 'attention', kernel: 'attention_decode_f16.wgsl', entry: 'attention_decode' },
        { op: 'ffn_gate_up', kernel: 'matmul_dequant_f32a.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_dequant_f32a.wgsl' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_dequant_f32a.wgsl' },
        { op: 'attention', kernel: 'attention_prefill_f16.wgsl' },
        { op: 'ffn_gate_up', kernel: 'matmul_dequant_f32a.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_dequant_f32a.wgsl' },
      ],
    },
    postLayer: [
      { op: 'lm_head', kernel: 'matmul_f16.wgsl' },
    ],
  };

  const gemma3Path = {
    id: 'gemma3-f16-fused-f16a-online',
    name: 'Gemma3 F16 Fused F16A Online',
    activationDtype: 'f16',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'gemv_subgroup_vec4_f16a.wgsl' },
        { op: 'attention', kernel: 'attention_decode_online_f16.wgsl' },
        { op: 'ffn_gate_up', kernel: 'gemv_subgroup_vec4_f16a.wgsl' },
        { op: 'ffn_down', kernel: 'gemv_subgroup_vec4_f16a.wgsl' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f16_tiled.wgsl' },
        { op: 'attention', kernel: 'attention_prefill_f16.wgsl' },
        { op: 'ffn_gate_up', kernel: 'matmul_f16_tiled.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_f16_tiled.wgsl' },
      ],
    },
    postLayer: [
      { op: 'lm_head', kernel: 'matmul_f16.wgsl' },
    ],
  };

  const gemma3F32WeightPath = {
    id: 'gemma3-q4k-dequant-f32w-f32a-online',
    name: 'Gemma3 Q4K Dequant F32W F32A Online',
    activationDtype: 'f32',
    decode: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f32.wgsl' },
        { op: 'attention', kernel: 'attention_decode_online_f16.wgsl' },
        { op: 'ffn_gate_up', kernel: 'matmul_f32.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_f32.wgsl' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'matmul_f32.wgsl' },
        { op: 'attention', kernel: 'attention_prefill_f16.wgsl' },
        { op: 'ffn_gate_up', kernel: 'matmul_f32.wgsl' },
        { op: 'ffn_down', kernel: 'matmul_f32.wgsl' },
      ],
    },
    postLayer: [
      { op: 'lm_head', kernel: 'matmul_f32.wgsl' },
    ],
  };

  const qwenLinearPath = {
    id: 'qwen-linear-prefill-q4',
    name: 'Qwen Linear Prefill Q4',
    activationDtype: 'f32',
    decode: {
      steps: [
        {
          op: 'q_proj',
          kernel: 'fused_matmul_q4_multicol_f16a.wgsl',
          entry: 'main_multicol_f16a',
          precision: { inputDtype: 'f16', outputDtype: 'f16' },
        },
        { op: 'o_proj', kernel: 'fused_matmul_q4.wgsl', entry: 'main_multicol' },
      ],
    },
    prefill: {
      steps: [
        { op: 'q_proj', kernel: 'fused_matmul_q4_batched.wgsl', entry: 'main' },
        { op: 'o_proj', kernel: 'fused_matmul_q4_batched.wgsl', entry: 'main' },
      ],
    },
    postLayer: [],
  };

  assert.equal(getKernelPathActivationDtype({ activationDtype: 'f16' }), 'f16');
  assert.equal(getKernelPathActivationDtype({}), null);
  assert.equal(getKernelPathOutputDtype({ outputDtype: 'f32' }), 'f32');
  assert.equal(getKernelPathOutputDtype({}), null);
  assert.equal(getKernelPathKVDtype({ kvDtype: 'f16', activationDtype: 'f32' }), 'f16');
  assert.equal(getKernelPathKVDtype({ activationDtype: 'f32' }), 'f32');
  assert.equal(getKernelPathKVDtype(null), null);

  assert.equal(resolveWeightRef('layers.{L}.attn.{L}.weight', 7), 'layers.7.attn.7.weight');

  const overridePath = {
    id: 'override-test',
    name: 'Override test',
    activationDtype: 'f16',
    decode: {
      steps: [{ op: 'decode_only', kernel: 'decode_kernel.wgsl' }],
    },
    prefill: {
      steps: [{ op: 'prefill_only', kernel: 'prefill_kernel.wgsl' }],
    },
    layerOverrides: [
      {
        layers: [1],
        steps: [{ op: 'override_step', kernel: 'override_kernel.wgsl' }],
      },
      {
        layers: [2],
        decode: {
          steps: [{ op: 'decode_override', kernel: 'decode_override_kernel.wgsl' }],
        },
        prefill: {
          steps: [{ op: 'prefill_override', kernel: 'prefill_override_kernel.wgsl' }],
        },
      },
    ],
    postLayer: [
      { op: 'lm_head_prefill', kernel: 'matmul.wgsl', constants: { mode: 'prefill' } },
      { op: 'lm_head', kernel: 'matmul.wgsl', constants: { mode: 'decode' } },
    ],
  };
  assert.equal(getLayerSteps(overridePath, 1, 'decode')[0].op, 'override_step');
  assert.equal(getLayerSteps(overridePath, 1, 'prefill')[0].op, 'override_step');
  assert.equal(getLayerSteps(overridePath, 2, 'decode')[0].op, 'decode_override');
  assert.equal(getLayerSteps(overridePath, 2, 'prefill')[0].op, 'prefill_override');
  assert.equal(getLayerSteps(overridePath, 0, 'prefill')[0].op, 'prefill_only');
  assert.equal(getLayerSteps(overridePath, 0, 'decode')[0].op, 'decode_only');

  const validationErrors = validateKernelPath({
    id: '',
    name: '',
    activationDtype: null,
    decode: {
      steps: [{ op: '', kernel: '' }],
    },
    prefill: {
      steps: [{ op: null, kernel: null }],
    },
    preLayer: [{ op: null, kernel: null }],
    postLayer: [{ op: null, kernel: null }],
    sampling: [{ op: null, kernel: null }],
  });
  assert.ok(validationErrors.length >= 8);
  assert.ok(validationErrors.some((entry) => entry.includes('Missing path id')));
  assert.ok(validationErrors.some((entry) => entry.includes('missing kernel')));

  assert.deepEqual(
    getKernelPathMatmulConstants('lm_head', 'prefill', 0, overridePath),
    { mode: 'prefill' }
  );
  assert.deepEqual(
    getKernelPathMatmulConstants('lm_head', 'decode', 0, overridePath),
    { mode: 'decode' }
  );
  assert.equal(getKernelPathMatmulConstants('missing_role', 'decode', 0, overridePath), null);

  assert.equal(getKernelPathAttentionVariant('decode', 0, { decode: { steps: [] } }), null);
  assert.equal(getKernelPathMatmulVariant('missing_role', 'decode', 0, { decode: { steps: [] } }), null);
  assert.equal(
    getKernelPathMatmulVariant('linear_qkv_proj', 'prefill', 0, qwenLinearPath),
    getKernelPathMatmulVariant('q_proj', 'prefill', 0, qwenLinearPath)
  );
  assert.equal(
    getKernelPathMatmulVariant('linear_z_proj', 'prefill', 0, qwenLinearPath),
    getKernelPathMatmulVariant('q_proj', 'prefill', 0, qwenLinearPath)
  );
  assert.equal(
    getKernelPathMatmulVariant('linear_out_proj', 'decode', 0, qwenLinearPath),
    getKernelPathMatmulVariant('o_proj', 'decode', 0, qwenLinearPath)
  );
  assert.equal(
    getKernelPathMatmulPrecision('linear_a_proj', 'decode', 0, qwenLinearPath)?.outputDtype,
    'f16'
  );
  assert.equal(
    getKernelPathMatmulPrecision('linear_b_proj', 'decode', 0, qwenLinearPath)?.outputDtype,
    'f16'
  );
  assert.equal(
    getKernelPathMatmulPrecision('linear_qkv_proj', 'decode', 0, qwenLinearPath)?.outputDtype,
    'f16'
  );
  assert.equal(
    getKernelPathMatmulVariant('linear_qkv_proj', 'decode', 0, qwenLinearPath),
    'q4_fused_multicol_f16a'
  );

  assert.equal(isKernelPathFusedQ4K(fusedPath), true);
  assert.equal(isKernelPathFusedQ4K(dequantPath), false);
  assert.equal(isKernelPathDequant(fusedPath), true);
  assert.equal(kernelPathRequiresF32MatmulWeights(gemma3Path), false);
  assert.equal(kernelPathRequiresF32MatmulWeights(gemma3F32WeightPath), true);
  assert.equal(isKernelPathDequant({ decode: { steps: [{ kernel: 'attention.wgsl' }] } }), false);

  const stats = getKernelPathStats(gemma3Path);
  assert.ok(stats.decodeSteps > 0);
  assert.ok(stats.prefillSteps > 0);
  assert.ok(stats.uniqueKernels > 0);
  assert.equal(typeof stats.hasLayerOverrides, 'boolean');
  assert.match(formatKernelPath(gemma3Path), /^gemma3-f16-fused-f16a-online:/);

  setActiveKernelPath(fusedPath, 'runtime', {
    mode: 'capability-aware',
    sourceScope: ['config', 'model'],
    onIncompatible: 'remap',
  });
  assert.equal(getActiveKernelPath(), fusedPath);
  assert.equal(getActiveKernelPathSource(), 'runtime');
  assert.equal(isActiveKernelPathFusedQ4K(), true);
  assert.equal(isActiveKernelPathDequant(), true);
  assert.deepEqual(
    getActiveKernelPathPolicy(),
    {
      mode: 'capability-aware',
      sourceScope: ['config', 'model'],
      allowSources: ['config', 'model'],
      onIncompatible: 'remap',
    }
  );

  assert.throws(
    () => setActiveKernelPath(fusedPath, 'runtime', {
      mode: 'capability-aware',
      sourceScope: ['runtime'],
      onIncompatible: 'remap',
    }),
    /does not accept legacy "runtime". Use "config"/
  );
  assert.throws(
    () => setActiveKernelPath(fusedPath, 'runtime', {
      mode: 'capability-aware',
      sourceScope: ['execution_v0'],
      onIncompatible: 'remap',
    }),
    /does not accept "execution-v0"/
  );
  assert.throws(
    () => setActiveKernelPath(null, 'runtime', ['not-a-policy']),
    /kernelPathPolicy must be an object/
  );

  assert.equal(getKernelPathStrict(), true);
} finally {
  setActiveKernelPath(originalPath, originalSource, originalPolicy);
}

console.log('kernel-path-loader-contract.test: ok');
