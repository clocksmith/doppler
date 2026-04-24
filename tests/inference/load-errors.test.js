import assert from 'node:assert/strict';
import test from 'node:test';

import { ERROR_CODES } from '../../src/errors/doppler-error.js';
import { annotateWeightLoadError, rewriteWeightLoadError } from '../../src/inference/pipelines/text/load-errors.js';

test('rewriteWeightLoadError preserves unrelated errors', () => {
  const original = new Error('manifest shard 3 missing');
  assert.equal(rewriteWeightLoadError(original, { modelId: 'gemma-4-31b' }), original);
});

test('annotateWeightLoadError records the failing tensor context', () => {
  const original = new Error('writeBuffer failed');
  const annotated = annotateWeightLoadError(original, {
    tensorName: 'model.embed_tokens.weight',
    tensorLoadStage: 'materializeTensorToGPU',
    tensorSizeBytes: 4096,
  });

  assert.equal(annotated, original);
  assert.deepEqual(annotated.details?.weightLoadFailure, {
    tensorName: 'model.embed_tokens.weight',
    tensorLoadStage: 'materializeTensorToGPU',
    tensorSizeBytes: 4096,
  });
});

test('rewriteWeightLoadError rewrites device lifecycle failures with explicit guidance', () => {
  const original = new Error('Device not initialized');
  original.details = {
    weightLoadFailure: {
      tensorName: 'model.embed_tokens.weight',
      tensorLoadStage: 'materializeTensorToGPU',
      tensorShardIndices: [0],
    },
  };
  const rewritten = rewriteWeightLoadError(original, {
    modelId: 'gemma-4-31b-it-text-q4k-ehf16-af32',
    deviceLossInfo: {
      message: 'current device lost',
      reason: 'destroyed',
      deviceEpoch: 7,
      timestampMs: 1234,
      adapterInfo: {
        vendor: 'amd',
        architecture: 'rdna-3',
        device: 'apux',
        description: 'test adapter',
      },
    },
  });

  assert.notEqual(rewritten, original);
  assert.equal(rewritten.code, ERROR_CODES.GPU_DEVICE_FAILED);
  assert.match(
    rewritten.message,
    /device lifecycle failure during loadWeights, not proof of VRAM-capacity exhaustion/i
  );
  assert.equal(rewritten.cause, original);
  assert.equal(rewritten.details?.loadPhase, 'loadWeights');
  assert.equal(
    rewritten.details?.modelId,
    'gemma-4-31b-it-text-q4k-ehf16-af32'
  );
  assert.equal(
    rewritten.details?.lifecycleFailure,
    'device_unavailable_during_weight_load'
  );
  assert.equal(
    rewritten.details?.weightLoadFailure?.tensorName,
    'model.embed_tokens.weight'
  );
  assert.equal(
    rewritten.details?.lastDeviceLoss?.reason,
    'destroyed'
  );
});
