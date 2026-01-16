import { describe, expect, it, vi } from 'vitest';

const baseCaps = {
  hasSubgroups: false,
  hasSubgroupsF16: false,
  hasF16: false,
  hasTimestampQuery: false,
  maxBufferSize: 0,
  maxWorkgroupSize: 256,
  maxWorkgroupStorageSize: 65536,
  adapterInfo: { vendor: 'test', architecture: 'test', device: 'test', description: '' },
};

async function loadMatmulWithCaps(caps) {
  vi.resetModules();
  vi.doMock('../../src/gpu/device.js', () => ({
    getDevice: () => null,
    getKernelCapabilities: () => ({ ...baseCaps, ...caps }),
  }));
  return import('../../src/gpu/kernels/matmul.js');
}

async function loadDequantWithCaps(caps) {
  vi.resetModules();
  vi.doMock('../../src/gpu/device.js', () => ({
    getDevice: () => null,
    getKernelCapabilities: () => ({ ...baseCaps, ...caps }),
  }));
  return import('../../src/gpu/kernels/dequant.js');
}

describe('kernel selection (mock profiles)', () => {
  it('selects f16 matmul when f16 is available', async () => {
    const matmul = await loadMatmulWithCaps({ hasF16: true });
    const variant = matmul.selectMatmulKernel({
      outputDtype: 'f16',
      aDtype: 'f16',
      bDtype: 'f16',
    });
    expect(variant).toBe('f16');
  });

  it('falls back to f32 matmul without f16 support', async () => {
    const matmul = await loadMatmulWithCaps({ hasF16: false });
    const variant = matmul.selectMatmulKernel({
      outputDtype: 'f16',
      aDtype: 'f16',
      bDtype: 'f16',
    });
    expect(variant).toBe('f32');
  });

  it('selects subgroup dequant when subgroups are available', async () => {
    const dequant = await loadDequantWithCaps({ hasSubgroups: true });
    const variant = dequant.selectDequantKernel({ outputDtype: 'f32', useVec4: true });
    expect(variant).toBe('subgroup_vec4');
  });

  it('falls back to shared dequant without subgroups', async () => {
    const dequant = await loadDequantWithCaps({ hasSubgroups: false });
    const variant = dequant.selectDequantKernel({ outputDtype: 'f32', useVec4: true });
    expect(variant).toBe('shared_vec4');
  });

  it('selects attention tier based on capabilities', async () => {
    const { resolveAttentionPlanForTest } = await import('../../src/gpu/kernels/attention.js');
    const plan = resolveAttentionPlanForTest(
      1,
      128,
      64,
      2,
      'f32',
      'f32',
      65536,
      { hasSubgroups: true, hasF16: false }
    );
    expect(plan.tier).toBe('subgroup');
    expect(plan.variant).toBe('decode_subgroup');
  });
});
