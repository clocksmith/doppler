import assert from 'node:assert/strict';

import { validateManifest } from '../../src/formats/rdrr/validation.js';
import {
  DOPPLER_LOWERING_MISSING,
  DOPPLER_LOWERING_REJECTED,
  findLowering,
  findLoweringOrThrow,
  isRejectionEntry,
  listSupportedBackends,
} from '../../src/formats/rdrr/lowerings.js';

function successEntry(overrides = {}) {
  return {
    kernelRef: 'fused_gemv',
    backend: 'webgpu-generic',
    targetDescriptorHash: 'sha256:td',
    frontendVersion: 'doe-frontend-0.1.0',
    tsirSemanticDigest: 'sha256:sem',
    tsirRealizationDigest: 'sha256:real',
    emitterDigest: 'sha256:emit',
    doeCompilerVersion: 'doe-0.1.0',
    exactnessClass: 'algorithm-exact',
    rejectionReasons: null,
    ...overrides,
  };
}

function rejectionEntry(overrides = {}) {
  return {
    kernelRef: 'fused_gemv',
    backend: 'wse3',
    targetDescriptorHash: null,
    frontendVersion: null,
    tsirSemanticDigest: null,
    tsirRealizationDigest: null,
    emitterDigest: null,
    doeCompilerVersion: null,
    exactnessClass: null,
    rejectionReasons: ['TSIR_PE_BUDGET_EXHAUSTED'],
    ...overrides,
  };
}

function baseManifest(integrityExtensionsOverrides = {}) {
  return {
    version: 1,
    modelId: 'test-model',
    modelType: 'transformer',
    quantization: 'q4k',
    hashAlgorithm: 'sha256',
    eos_token_id: 1,
    architecture: {
      numLayers: 1,
      hiddenSize: 64,
      intermediateSize: 128,
      numAttentionHeads: 1,
      numKeyValueHeads: 1,
      headDim: 64,
      vocabSize: 1000,
      maxSeqLen: 1024,
    },
    shards: [{ index: 0, size: 1, hash: 'a'.repeat(64), filename: 's0.bin', offset: 0 }],
    totalSize: 1,
    tensors: { w: { role: 'other' } },
    integrityExtensions: {
      contractVersion: 1,
      blockMerkle: { blockSize: 256, roots: { w: 'sha256:x' } },
      ...integrityExtensionsOverrides,
    },
    inference: {},
  };
}

// =========================================================================
// Schema shape — absent lowerings section is fine
// =========================================================================

{
  const manifest = baseManifest();
  const result = validateManifest(manifest);
  const loweringErrors = result.errors.filter((e) => e.includes('lowerings'));
  assert.deepEqual(loweringErrors, []);
}

// =========================================================================
// Schema shape — well-formed success entry passes shape checks
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [successEntry()] },
  });
  const result = validateManifest(manifest);
  const loweringErrors = result.errors.filter((e) => e.includes('lowerings'));
  assert.deepEqual(loweringErrors, [], `unexpected errors: ${loweringErrors.join(', ')}`);
}

// =========================================================================
// Schema shape — well-formed rejection entry passes shape checks
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [rejectionEntry()] },
  });
  const result = validateManifest(manifest);
  const loweringErrors = result.errors.filter((e) => e.includes('lowerings'));
  assert.deepEqual(loweringErrors, []);
}

// =========================================================================
// Schema shape — bad contractVersion rejected
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 2, entries: [] },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('lowerings.contractVersion')),
    'expected error about lowerings.contractVersion'
  );
}

// =========================================================================
// Schema shape — entries must be array
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: 'nope' },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('lowerings.entries')),
    'expected error about lowerings.entries'
  );
}

// =========================================================================
// Schema shape — missing kernelRef rejected
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [successEntry({ kernelRef: '' })] },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('kernelRef')),
    'expected error about kernelRef'
  );
}

// =========================================================================
// Schema shape — undefined nullable field rejected (must be explicitly null)
// =========================================================================

{
  const entry = successEntry();
  delete entry.doeCompilerVersion;
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [entry] },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('doeCompilerVersion is required')),
    'expected error requiring explicit null for nullable field'
  );
}

// =========================================================================
// Schema shape — invalid exactnessClass rejected
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: {
      contractVersion: 1,
      entries: [successEntry({ exactnessClass: 'approximate' })],
    },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('exactnessClass')),
    'expected error about exactnessClass'
  );
}

// =========================================================================
// Schema shape — mixed state (rejection AND digests) rejected
// =========================================================================

{
  const bad = successEntry({ rejectionReasons: ['TSIR_TARGET_UNFIT'] });
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [bad] },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('both rejectionReasons and digests')),
    'expected error about mixed rejection+digest state'
  );
}

// =========================================================================
// Schema shape — partial digests (some null, some set) rejected
// =========================================================================

{
  const bad = successEntry({ tsirSemanticDigest: null });
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [bad] },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('inconsistent digests')),
    'expected error about inconsistent digest state'
  );
}

// =========================================================================
// Schema shape — duplicate (kernelRef, backend) rejected
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: {
      contractVersion: 1,
      entries: [successEntry(), successEntry()],
    },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('duplicates')),
    'expected error about duplicate (kernelRef, backend) pair'
  );
}

// =========================================================================
// Schema shape — empty rejectionReasons array rejected
// =========================================================================

{
  const bad = rejectionEntry({ rejectionReasons: [] });
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [bad] },
  });
  const result = validateManifest(manifest);
  assert.ok(
    result.errors.some((e) => e.includes('rejectionReasons')),
    'expected error about empty rejectionReasons'
  );
}

// =========================================================================
// findLowering — returns entry when present
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [successEntry()] },
  });
  const entry = findLowering(manifest, 'fused_gemv', 'webgpu-generic');
  assert.ok(entry);
  assert.equal(entry.kernelRef, 'fused_gemv');
}

// =========================================================================
// findLowering — returns null when absent
// =========================================================================

{
  const manifest = baseManifest();
  assert.equal(findLowering(manifest, 'fused_gemv', 'webgpu-generic'), null);
}

// =========================================================================
// findLoweringOrThrow — success path returns entry
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [successEntry()] },
  });
  const entry = findLoweringOrThrow(manifest, 'fused_gemv', 'webgpu-generic');
  assert.equal(entry.exactnessClass, 'algorithm-exact');
}

// =========================================================================
// findLoweringOrThrow — missing throws DOPPLER_LOWERING_MISSING
// =========================================================================

{
  const manifest = baseManifest();
  let caught = null;
  try {
    findLoweringOrThrow(manifest, 'fused_gemv', 'webgpu-generic');
  } catch (error) {
    caught = error;
  }
  assert.ok(caught);
  assert.equal(caught.code, DOPPLER_LOWERING_MISSING);
  assert.equal(caught.kernelRef, 'fused_gemv');
  assert.equal(caught.backend, 'webgpu-generic');
}

// =========================================================================
// findLoweringOrThrow — rejection throws DOPPLER_LOWERING_REJECTED
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: { contractVersion: 1, entries: [rejectionEntry()] },
  });
  let caught = null;
  try {
    findLoweringOrThrow(manifest, 'fused_gemv', 'wse3');
  } catch (error) {
    caught = error;
  }
  assert.ok(caught);
  assert.equal(caught.code, DOPPLER_LOWERING_REJECTED);
  assert.deepEqual(caught.rejectionReasons, ['TSIR_PE_BUDGET_EXHAUSTED']);
}

// =========================================================================
// isRejectionEntry
// =========================================================================

{
  assert.equal(isRejectionEntry(successEntry()), false);
  assert.equal(isRejectionEntry(rejectionEntry()), true);
  assert.equal(isRejectionEntry(null), false);
  assert.equal(isRejectionEntry({ rejectionReasons: [] }), false);
}

// =========================================================================
// listSupportedBackends — backend with full coverage included
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: {
      contractVersion: 1,
      entries: [
        successEntry({ kernelRef: 'fused_gemv', backend: 'webgpu-generic' }),
        successEntry({ kernelRef: 'rmsnorm', backend: 'webgpu-generic' }),
      ],
    },
  });
  const supported = listSupportedBackends(manifest, ['fused_gemv', 'rmsnorm']);
  assert.deepEqual(supported, ['webgpu-generic']);
}

// =========================================================================
// listSupportedBackends — backend with partial coverage excluded
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: {
      contractVersion: 1,
      entries: [successEntry({ kernelRef: 'fused_gemv', backend: 'webgpu-generic' })],
    },
  });
  const supported = listSupportedBackends(manifest, ['fused_gemv', 'rmsnorm']);
  assert.deepEqual(supported, []);
}

// =========================================================================
// listSupportedBackends — any rejection excludes the whole backend
// =========================================================================

{
  const manifest = baseManifest({
    lowerings: {
      contractVersion: 1,
      entries: [
        successEntry({ kernelRef: 'fused_gemv', backend: 'wse3' }),
        rejectionEntry({ kernelRef: 'rmsnorm', backend: 'wse3' }),
      ],
    },
  });
  const supported = listSupportedBackends(manifest, ['fused_gemv', 'rmsnorm']);
  assert.deepEqual(supported, []);
}

console.log('rdrr-lowerings: all assertions passed');
