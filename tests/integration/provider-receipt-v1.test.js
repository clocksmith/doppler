import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { buildProviderReceiptV1 } from '../../src/client/receipt.js';

describe('buildProviderReceiptV1', () => {
  it('returns a receipt with version and id', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.equal(receipt.receiptVersion, 'doppler_provider_receipt_v1');
    assert.ok(typeof receipt.receiptId === 'string');
    assert.ok(receipt.receiptId.length > 0);
  });

  it('populates source and policyMode', () => {
    const receipt = buildProviderReceiptV1({
      source: 'fallback',
      policyMode: 'prefer-local',
      totalDurationMs: 200,
    });
    assert.equal(receipt.source, 'fallback');
    assert.equal(receipt.policyMode, 'prefer-local');
  });

  it('includes policyId when provided', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      policyId: 'columbo_local_protected_v1',
      totalDurationMs: 50,
    });
    assert.equal(receipt.policyId, 'columbo_local_protected_v1');
  });

  it('defaults policyId to null when not provided', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 50,
    });
    assert.equal(receipt.policyId, null);
  });

  it('populates model fields', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      model: { id: 'gemma-4-e2b', hash: 'abc123', fallbackId: null },
      totalDurationMs: 100,
    });
    assert.equal(receipt.model.id, 'gemma-4-e2b');
    assert.equal(receipt.model.hash, 'abc123');
    assert.equal(receipt.model.fallbackId, null);
  });

  it('defaults model to empty strings and nulls', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.equal(receipt.model.id, '');
    assert.equal(receipt.model.hash, null);
    assert.equal(receipt.model.fallbackId, null);
  });

  it('builds device snapshot from kernelCapabilities', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      kernelCapabilities: {
        adapterInfo: { vendor: 'apple', architecture: 'gpu', device: 'm1', description: 'Apple M1' },
        hasF16: true,
        hasSubgroups: false,
        maxBufferSize: 1073741824,
        submitProbeMs: 0.5,
      },
      deviceEpoch: 3,
      totalDurationMs: 100,
    });
    assert.ok(receipt.device);
    assert.equal(receipt.device.vendor, 'apple');
    assert.equal(receipt.device.hasF16, true);
    assert.equal(receipt.device.hasSubgroups, false);
    assert.equal(receipt.device.maxBufferSize, 1073741824);
    assert.equal(receipt.device.submitProbeMs, 0.5);
    assert.equal(receipt.device.deviceEpoch, 3);
  });

  it('device is null when no device info provided', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.equal(receipt.device, null);
  });

  it('includes failure info with fine-grained class and extended fields', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'prefer-local',
      failure: {
        failureClass: 'gpu_oom',
        failureCode: 'DOPPLER_GPU_OOM',
        stage: 'prefill',
        surface: 'webgpu',
        device: null,
        modelId: 'gemma-3-270m',
        runtimeProfile: 'approved_discrete',
        kernelPathId: null,
        isSimulated: true,
        message: 'out of memory',
      },
      totalDurationMs: 100,
    });
    assert.ok(receipt.failure);
    assert.equal(receipt.failure.failureClass, 'gpu_oom');
    assert.equal(receipt.failure.failureCode, 'DOPPLER_GPU_OOM');
    assert.equal(receipt.failure.modelId, 'gemma-3-270m');
    assert.equal(receipt.failure.runtimeProfile, 'approved_discrete');
    assert.equal(receipt.failure.kernelPathId, null);
    assert.equal(receipt.failure.device, null);
    assert.equal(receipt.failure.isSimulated, true);
  });

  it('defaults new FailureRecord fields to null when unspecified', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      failure: {
        failureClass: 'gpu_device_lost',
        failureCode: 'DOPPLER_GPU_DEVICE_LOST',
        stage: 'decode',
        surface: 'webgpu',
        isSimulated: false,
        message: 'device lost',
      },
      totalDurationMs: 100,
    });
    assert.equal(receipt.failure.device, null);
    assert.equal(receipt.failure.modelId, null);
    assert.equal(receipt.failure.runtimeProfile, null);
    assert.equal(receipt.failure.kernelPathId, null);
  });

  it('failure is null when not provided', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.equal(receipt.failure, null);
  });

  it('includes fallback decision', () => {
    const receipt = buildProviderReceiptV1({
      source: 'fallback',
      policyMode: 'prefer-local',
      fallbackDecision: {
        reason: 'DOPPLER_GPU_OOM',
        eligible: true,
        executed: true,
        deniedReason: null,
      },
      totalDurationMs: 300,
    });
    assert.ok(receipt.fallbackDecision);
    assert.equal(receipt.fallbackDecision.eligible, true);
    assert.equal(receipt.fallbackDecision.executed, true);
  });

  it('includes timing fields', () => {
    const receipt = buildProviderReceiptV1({
      source: 'fallback',
      policyMode: 'prefer-local',
      localDurationMs: 50,
      fallbackDurationMs: 200,
      totalDurationMs: 250,
    });
    assert.equal(receipt.localDurationMs, 50);
    assert.equal(receipt.fallbackDurationMs, 200);
    assert.equal(receipt.totalDurationMs, 250);
  });

  it('defaults timing nulls correctly', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.equal(receipt.localDurationMs, null);
    assert.equal(receipt.fallbackDurationMs, null);
  });

  it('includes timestamp', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.ok(typeof receipt.timestamp === 'string');
    // Should be a valid ISO date
    assert.ok(!isNaN(Date.parse(receipt.timestamp)));
  });

  it('defaults diagnoseArtifactRef to null', () => {
    const receipt = buildProviderReceiptV1({
      source: 'local',
      policyMode: 'local-only',
      totalDurationMs: 100,
    });
    assert.equal(receipt.diagnoseArtifactRef, null);
  });
});
