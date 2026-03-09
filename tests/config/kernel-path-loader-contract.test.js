import assert from 'node:assert/strict';

import {
  formatKernelPath,
  getActiveKernelPath,
  getActiveKernelPathPolicy,
  getActiveKernelPathSource,
  getKernelPath,
  getKernelPathActivationDtype,
  getKernelPathAttentionVariant,
  getKernelPathKVDtype,
  getKernelPathMatmulConstants,
  getKernelPathMatmulVariant,
  getKernelPathOutputDtype,
  getKernelPathStats,
  getKernelPathStrict,
  getLayerSteps,
  isActiveKernelPathDequant,
  isActiveKernelPathFusedQ4K,
  isKernelPathDequant,
  isKernelPathFusedQ4K,
  listKernelPaths,
  resolveKernelPath,
  resolveWeightRef,
  setActiveKernelPath,
  validateKernelPath,
} from '../../src/config/kernel-path-loader.js';

const originalPath = getActiveKernelPath();
const originalSource = getActiveKernelPathSource();
const originalPolicy = getActiveKernelPathPolicy();

try {
  const all = listKernelPaths();
  assert.ok(Array.isArray(all));
  assert.ok(all.length > 0);
  assert.ok(all.includes('gemma2-q4k-dequant-f32a-nosubgroups'));
  assert.ok(all.includes('gemma2-q4k-dequant-f32a'));

  assert.equal(getKernelPath('missing-kernel-path-id'), null);
  assert.throws(
    () => resolveKernelPath('missing-kernel-path-id'),
    /Unknown kernel path/
  );
  assert.equal(resolveKernelPath(null), null);

  const fusedPath = resolveKernelPath('gemma2-q4k-fused-f32a');
  const dequantPath = resolveKernelPath('gemma2-q4k-dequant-f32a');
  const canonicalDequantPath = resolveKernelPath('gemma2-q4k-dequant-f32a-nosubgroups');
  const gemma3Path = resolveKernelPath('gemma3-f16-fused-f16a-online');
  assert.ok(fusedPath);
  assert.ok(dequantPath);
  assert.ok(canonicalDequantPath);
  assert.ok(gemma3Path);
  assert.equal(dequantPath.id, 'gemma2-q4k-dequant-f32a-nosubgroups');
  assert.equal(canonicalDequantPath.id, 'gemma2-q4k-dequant-f32a-nosubgroups');

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
    ],
    postLayer: [
      { op: 'lm_head_prefill', kernel: 'matmul.wgsl', constants: { mode: 'prefill' } },
      { op: 'lm_head', kernel: 'matmul.wgsl', constants: { mode: 'decode' } },
    ],
  };
  assert.equal(getLayerSteps(overridePath, 1, 'decode')[0].op, 'override_step');
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

  assert.equal(getKernelPathMatmulVariant('q_proj', 'decode', 0, gemma3Path), 'gemv_subgroup_vec4_f16a');
  assert.equal(getKernelPathMatmulVariant('q_proj', 'prefill', 0, gemma3Path), 'f16_tiled');
  assert.equal(getKernelPathAttentionVariant('decode', 0, gemma3Path), 'decode_online_f16');
  assert.equal(getKernelPathAttentionVariant('decode', 0, { decode: { steps: [] } }), null);
  assert.equal(getKernelPathMatmulVariant('missing_role', 'decode', 0, { decode: { steps: [] } }), null);

  assert.deepEqual(
    getKernelPathMatmulConstants('lm_head', 'prefill', 0, overridePath),
    { mode: 'prefill' }
  );
  assert.deepEqual(
    getKernelPathMatmulConstants('lm_head', 'decode', 0, overridePath),
    { mode: 'decode' }
  );
  assert.equal(getKernelPathMatmulConstants('missing_role', 'decode', 0, overridePath), null);

  assert.equal(isKernelPathFusedQ4K(fusedPath), true);
  assert.equal(isKernelPathFusedQ4K(dequantPath), false);
  assert.equal(isKernelPathDequant(fusedPath), true);
  assert.equal(isKernelPathDequant({ decode: { steps: [{ kernel: 'attention.wgsl' }] } }), false);

  const stats = getKernelPathStats(gemma3Path);
  assert.ok(stats.decodeSteps > 0);
  assert.ok(stats.prefillSteps > 0);
  assert.ok(stats.uniqueKernels > 0);
  assert.equal(typeof stats.hasLayerOverrides, 'boolean');
  assert.match(formatKernelPath(gemma3Path), /^gemma3-f16-fused-f16a-online:/);

  setActiveKernelPath(fusedPath, 'runtime', {
    mode: 'capability-aware',
    sourceScope: ['config', 'execution-v0'],
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
      sourceScope: ['config', 'execution-v0'],
      allowSources: ['config', 'execution-v0'],
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
    /does not accept legacy "execution_v0". Use "execution-v0"/
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
