import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import path from 'node:path';

import {
  buildGemma4PreflightReceipt,
  buildGemma4PreflightReceiptPath,
  formatReceiptTimestamp,
  parseArgs,
  serializeReceipt,
} from '../../tools/generate-gemma4-preflight-receipts.js';

const embeddingName = 'model.language_model.embed_tokens.weight';
const manifest = {
  modelId: 'gemma-4-e2b-unit',
  inference: {
    largeWeights: {
      gpuResidentOverrides: [embeddingName],
    },
    execution: {
      kernels: {
        embed: {
          kernel: 'gather_split8_f16_vec4_f32_out.wgsl',
          entry: 'gather_vec4_f32_out',
          digest: 'sha256:18dce4731aa3c3cbde6a969cc54d63de4e545e7dce43204774323b6bfffc5c41',
        },
      },
    },
  },
  tensors: {
    [embeddingName]: {
      shape: [14, 16],
      dtype: 'F16',
      role: 'embedding',
      group: 'embed',
      size: 448,
    },
  },
};

const manifestRaw = `${JSON.stringify(manifest, null, 2)}\n`;
const fakeDevice = {
  limits: {
    maxStorageBufferBindingSize: 64,
    maxBufferSize: 64,
    maxStorageBuffersPerShaderStage: 10,
  },
};
const capabilities = {
  hasF16: true,
  hasSubgroups: true,
  adapterInfo: {
    vendor: 'unit-vendor',
    architecture: 'unit-arch',
    device: 'unit-device',
    description: 'unit adapter',
  },
};

{
  const receipt = buildGemma4PreflightReceipt({
    manifest,
    manifestPath: 'models/local/gemma-4-e2b-unit/manifest.json',
    manifestRaw,
    generatedAt: '2026-06-21T12:34:56.000Z',
    capabilities,
    device: fakeDevice,
    largeWeightMaxBytes: 64,
    limitError: null,
  });

  assert.equal(receipt.receiptVersion, 'doppler_gemma4_preflight_receipt_v1');
  assert.equal(receipt.schemaVersion, 1);
  assert.equal(receipt.status, 'pass');
  assert.equal(receipt.modelId, 'gemma-4-e2b-unit');
  assert.equal(receipt.manifest.path, 'models/local/gemma-4-e2b-unit/manifest.json');
  assert.equal(
    receipt.manifest.sha256,
    `sha256:${crypto.createHash('sha256').update(manifestRaw).digest('hex')}`
  );
  assert.deepEqual(receipt.adapterInfo, capabilities.adapterInfo);
  assert.deepEqual(receipt.capabilities, {
    hasF16: true,
    hasSubgroups: true,
  });
  assert.deepEqual(receipt.deviceLimits, {
    maxStorageBufferBindingSize: 64,
    maxBufferSize: 64,
    maxStorageBuffersPerShaderStage: 10,
  });
  assert.deepEqual(receipt.embedding, {
    tensorName: embeddingName,
    dtype: 'F16',
    shape: [14, 16],
    tensorSizeBytes: 448,
    kernel: manifest.inference.execution.kernels.embed,
  });
  assert.deepEqual(receipt.preflight, {
    ok: true,
    splitKernelExpected: true,
    activeSplitKernelMaxSections: 8,
    maxSplitEmbeddingSections: 8,
    requiredSplitSections: 7,
    requiredStorageBuffers: 10,
    largeWeightMaxBytes: 64,
    rowsPerSplitSection: 2,
  });
  assert.equal(formatReceiptTimestamp(receipt.generatedAt), '2026-06-21T123456Z');
  assert.equal(
    buildGemma4PreflightReceiptPath({
      outputRoot: path.resolve('reports/gemma4-preflight'),
      modelId: receipt.modelId,
      generatedAt: receipt.generatedAt,
    }),
    path.resolve('reports/gemma4-preflight/gemma-4-e2b-unit/2026-06-21T123456Z.preflight.json')
  );
  assert.equal(serializeReceipt(receipt).endsWith('\n'), true);
}

{
  const limitError = new Error('unit preflight failure');
  limitError.details = {
    weightLoadFailure: {
      tensorName: embeddingName,
      tensorLoadStage: 'gpuResidentEmbeddingLimitPreflight',
      deviceLimitFailure: {
        kind: 'gpu_resident_embedding_exceeds_device_limit',
      },
    },
  };
  const receipt = buildGemma4PreflightReceipt({
    manifest,
    manifestPath: 'models/local/gemma-4-e2b-unit/manifest.json',
    manifestRaw,
    generatedAt: '2026-06-21T12:34:56.000Z',
    capabilities,
    device: fakeDevice,
    largeWeightMaxBytes: 64,
    limitError,
  });
  assert.equal(receipt.status, 'diagnostic');
  assert.equal(receipt.preflight.ok, false);
  assert.deepEqual(receipt.failure, limitError.details.weightLoadFailure);
}

assert.throws(
  () => parseArgs(['--manifest']),
  /Missing value for --manifest/
);

assert.throws(
  () => parseArgs(['--check', '--manifest', 'models/local/unit/manifest.json']),
  /--check reads preflight receipts from the target matrix/
);

console.log('gemma4-preflight-receipts.test: ok');
