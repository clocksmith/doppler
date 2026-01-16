

import { describe, it, expect } from 'vitest';
import {
  crossProduct,
  configName,
  QUANT_PATH_DIMS,
  isValidQuantConfig,
} from '../harness/test-matrix.js';

// ============================================================================
// Pure Selection Logic (extracted for testability)
// ============================================================================


function selectMatmulPath(config) {
  const {
    weightQuant,
    activationDtype = 'f32',
    hasSubgroups = true,
    hasF16 = true,
    M = 1,
    N = 1024,
    preferFused = true,
  } = config;

  const isDecode = M === 1;

  // Q4K fused path - requires subgroups
  if (weightQuant === 'q4k' && hasSubgroups && preferFused) {
    if (N > 8192 && isDecode) {
      return 'q4_fused_multicol';
    }
    return isDecode ? 'q4_fused' : 'q4_fused_batched';
  }

  // Q4K dequant path (fallback when fused disabled or no subgroups)
  if (weightQuant === 'q4k') {
    return 'dequant_then_matmul';
  }

  // F16 weights
  if (weightQuant === 'f16') {
    if (activationDtype === 'f16' && hasF16) {
      return 'f16';
    }
    if (activationDtype === 'f32' && hasF16) {
      return 'f16w_f32a';
    }
    return 'f32'; // Fallback if no F16 support
  }

  // Other quant types always use dequant path
  if (['q6k', 'q8', 'mxfp4'].includes(weightQuant)) {
    return 'dequant_then_matmul';
  }

  // F32 weights
  return 'f32';
}


function selectDequantVariant(config) {
  const {
    quantType = 'q4k',
    outputDtype = 'f32',
    useVec4 = true,
    hasSubgroups = true,
    hasF16 = true,
    isExpert = false,
  } = config;

  // Special quant formats
  if (quantType === 'q6k' && hasF16) {
    return 'q6k_f16out';
  }
  if (quantType === 'q8_0' && hasF16) {
    return 'q8_0_f16out';
  }
  if (quantType === 'mxfp4') {
    if (isExpert) return 'mxfp4_expert';
    return useVec4 ? 'mxfp4_vec4' : 'mxfp4';
  }

  // Standard Q4K dequant
  const wantsF16Out = outputDtype === 'f16' && hasF16;

  if (hasSubgroups) {
    if (wantsF16Out) {
      return useVec4 ? 'subgroup_vec4_f16out' : 'subgroup_f16out';
    }
    return useVec4 ? 'subgroup_vec4' : 'subgroup';
  }

  if (wantsF16Out) {
    return useVec4 ? 'shared_vec4_f16out' : 'shared_f16out';
  }
  return useVec4 ? 'shared_vec4' : 'shared';
}


function calculateFusedSavings(config) {
  const { M, N, K } = config;

  // Dequant path: Read quantized, write full, then matmul reads full again
  const Q4K_BYTES_PER_256 = 144; // Q4_K block size
  const numBlocks = Math.ceil(K / 256) * N;
  const quantizedBytes = numBlocks * Q4K_BYTES_PER_256;
  const dequantOutputBytes = K * N * 4; // F32 dequantized
  const matmulReadA = M * K * 4;
  const matmulReadB = K * N * 4;
  const matmulOutput = M * N * 4;
  const dequantPathBytes = quantizedBytes + dequantOutputBytes + matmulReadA + matmulReadB + matmulOutput;

  // Fused path: Read quantized directly, no intermediate
  const fusedPathBytes = quantizedBytes + matmulReadA + matmulOutput;

  const savingsBytes = dequantPathBytes - fusedPathBytes;
  const savingsPct = (savingsBytes / dequantPathBytes) * 100;

  return {
    dequantPathBytes,
    fusedPathBytes,
    savingsBytes,
    savingsPct,
  };
}


function selectEmbeddingPath(config) {
  const { embeddingQuant, hasF16 = true } = config;

  if (embeddingQuant === 'f16' && hasF16) {
    return 'gather_f16';
  }
  if (embeddingQuant === 'f32') {
    return 'gather_f32';
  }
  return 'gather_f32'; // Default
}


function selectKVDtype(config) {
  const { kvDtype = 'f32', hasF16 = true } = config;

  if (kvDtype === 'f16' && hasF16) {
    return 'f16';
  }
  return 'f32';
}

// ============================================================================
// Tests
// ============================================================================

describe('Matmul Path Selection', () => {
  describe('Q4K paths', () => {
    it('should select fused path when subgroups available', () => {
      const path = selectMatmulPath({
        weightQuant: 'q4k',
        hasSubgroups: true,
        M: 1,
      });
      expect(path).toBe('q4_fused');
    });

    it('should select multicol for large vocab', () => {
      const path = selectMatmulPath({
        weightQuant: 'q4k',
        hasSubgroups: true,
        M: 1,
        N: 32000, // Large vocab
      });
      expect(path).toBe('q4_fused_multicol');
    });

    it('should select batched for prefill', () => {
      const path = selectMatmulPath({
        weightQuant: 'q4k',
        hasSubgroups: true,
        M: 64, // Prefill batch
      });
      expect(path).toBe('q4_fused_batched');
    });

    it('should fall back to dequant when no subgroups', () => {
      const path = selectMatmulPath({
        weightQuant: 'q4k',
        hasSubgroups: false,
      });
      expect(path).toBe('dequant_then_matmul');
    });

    it('should fall back to dequant when fused disabled', () => {
      const path = selectMatmulPath({
        weightQuant: 'q4k',
        hasSubgroups: true,
        preferFused: false,
      });
      expect(path).toBe('dequant_then_matmul');
    });
  });

  describe('F16 paths', () => {
    it('should select f16 for pure F16', () => {
      const path = selectMatmulPath({
        weightQuant: 'f16',
        activationDtype: 'f16',
        hasF16: true,
      });
      expect(path).toBe('f16');
    });

    it('should select mixed precision for F16 weights + F32 activations', () => {
      const path = selectMatmulPath({
        weightQuant: 'f16',
        activationDtype: 'f32',
        hasF16: true,
      });
      expect(path).toBe('f16w_f32a');
    });

    it('should fall back to f32 when no F16 support', () => {
      const path = selectMatmulPath({
        weightQuant: 'f16',
        hasF16: false,
      });
      expect(path).toBe('f32');
    });
  });

  describe('Other quant types', () => {
    it('should use dequant path for Q6K', () => {
      const path = selectMatmulPath({ weightQuant: 'q6k' });
      expect(path).toBe('dequant_then_matmul');
    });

    it('should use dequant path for Q8', () => {
      const path = selectMatmulPath({ weightQuant: 'q8' });
      expect(path).toBe('dequant_then_matmul');
    });

    it('should use dequant path for MXFP4', () => {
      const path = selectMatmulPath({ weightQuant: 'mxfp4' });
      expect(path).toBe('dequant_then_matmul');
    });
  });

  // Parametrized tests
  describe('Matmul path matrix', () => {
    const configs = crossProduct({
      weightQuant: ['f32', 'f16', 'q4k', 'q8'],
      hasSubgroups: [true, false],
      hasF16: [true, false],
      M: [1, 64],
    });

    for (const config of configs) {
      it(`[${configName(config)}]`, () => {
        const path = selectMatmulPath(config);
        expect(path).toBeDefined();
        expect(typeof path).toBe('string');

        // Verify constraints
        if (config.weightQuant === 'q4k') {
          if (config.hasSubgroups) {
            expect(path).toMatch(/q4_fused|dequant/);
          } else {
            expect(path).toBe('dequant_then_matmul');
          }
        }
      });
    }
  });
});

describe('Dequant Variant Selection', () => {
  describe('Q4K variants', () => {
    it('should select subgroup_vec4 with subgroups', () => {
      const variant = selectDequantVariant({
        quantType: 'q4k',
        hasSubgroups: true,
        useVec4: true,
      });
      expect(variant).toBe('subgroup_vec4');
    });

    it('should select shared_vec4 without subgroups', () => {
      const variant = selectDequantVariant({
        quantType: 'q4k',
        hasSubgroups: false,
        useVec4: true,
      });
      expect(variant).toBe('shared_vec4');
    });

    it('should select f16out variants for F16 output', () => {
      const variant = selectDequantVariant({
        quantType: 'q4k',
        outputDtype: 'f16',
        hasSubgroups: true,
        hasF16: true,
      });
      expect(variant).toBe('subgroup_vec4_f16out');
    });
  });

  describe('Special quant types', () => {
    it('should select q6k_f16out for Q6K', () => {
      const variant = selectDequantVariant({
        quantType: 'q6k',
        hasF16: true,
      });
      expect(variant).toBe('q6k_f16out');
    });

    it('should select q8_0_f16out for Q8', () => {
      const variant = selectDequantVariant({
        quantType: 'q8_0',
        hasF16: true,
      });
      expect(variant).toBe('q8_0_f16out');
    });

    it('should select mxfp4_vec4 for MXFP4', () => {
      const variant = selectDequantVariant({
        quantType: 'mxfp4',
        useVec4: true,
      });
      expect(variant).toBe('mxfp4_vec4');
    });

    it('should select mxfp4_expert for MoE experts', () => {
      const variant = selectDequantVariant({
        quantType: 'mxfp4',
        isExpert: true,
      });
      expect(variant).toBe('mxfp4_expert');
    });
  });

  // Parametrized tests
  describe('Dequant variant matrix', () => {
    const configs = crossProduct({
      quantType: ['q4k', 'q6k', 'q8_0', 'mxfp4'],
      outputDtype: ['f32', 'f16'],
      useVec4: [true, false],
      hasSubgroups: [true, false],
      hasF16: [true, false],
    });

    for (const config of configs) {
      // Skip invalid: f16 output without F16 support
      if (config.outputDtype === 'f16' && !config.hasF16) continue;

      it(`[${configName(config)}]`, () => {
        const variant = selectDequantVariant(config);
        expect(variant).toBeDefined();
        expect(typeof variant).toBe('string');

        // Verify naming patterns
        if (config.quantType === 'q4k') {
          if (config.hasSubgroups) {
            expect(variant).toMatch(/subgroup/);
          } else {
            expect(variant).toMatch(/shared/);
          }
        }
      });
    }
  });
});

describe('Fused vs Dequant Bandwidth', () => {
  it('should show significant savings for fused path', () => {
    const savings = calculateFusedSavings({
      M: 1,
      N: 4096,
      K: 4096,
    });

    // Fused path should save substantial bandwidth
    expect(savings.savingsPct).toBeGreaterThan(30);
  });

  it('should scale savings with matrix size', () => {
    const small = calculateFusedSavings({ M: 1, N: 1024, K: 1024 });
    const large = calculateFusedSavings({ M: 1, N: 4096, K: 4096 });

    // Larger matrices save more absolute bytes
    expect(large.savingsBytes).toBeGreaterThan(small.savingsBytes);
  });

  // Parametrized tests
  describe('Savings matrix', () => {
    const configs = crossProduct({
      M: [1, 8, 64],
      N: [1024, 4096],
      K: [1024, 4096],
    });

    for (const config of configs) {
      it(`[M=${config.M}, N=${config.N}, K=${config.K}]`, () => {
        const savings = calculateFusedSavings(config);

        expect(savings.fusedPathBytes).toBeLessThan(savings.dequantPathBytes);
        expect(savings.savingsBytes).toBeGreaterThan(0);
        expect(savings.savingsPct).toBeGreaterThan(0);
      });
    }
  });
});

describe('Embedding Path Selection', () => {
  it('should select gather_f16 for F16 embeddings', () => {
    const path = selectEmbeddingPath({ embeddingQuant: 'f16', hasF16: true });
    expect(path).toBe('gather_f16');
  });

  it('should select gather_f32 for F32 embeddings', () => {
    const path = selectEmbeddingPath({ embeddingQuant: 'f32' });
    expect(path).toBe('gather_f32');
  });

  it('should fall back to f32 when no F16 support', () => {
    const path = selectEmbeddingPath({ embeddingQuant: 'f16', hasF16: false });
    expect(path).toBe('gather_f32');
  });
});

describe('KV Cache Dtype Selection', () => {
  it('should select f16 when supported', () => {
    const dtype = selectKVDtype({ kvDtype: 'f16', hasF16: true });
    expect(dtype).toBe('f16');
  });

  it('should fall back to f32 when F16 not supported', () => {
    const dtype = selectKVDtype({ kvDtype: 'f16', hasF16: false });
    expect(dtype).toBe('f32');
  });
});

describe('Quantization Config Validation', () => {
  it('should reject fused Q4K with non-Q4K weights', () => {
    expect(isValidQuantConfig({ fusedQ4K: true, weightQuant: 'f16' })).toBe(false);
  });

  it('should accept fused Q4K with Q4K weights', () => {
    expect(isValidQuantConfig({ fusedQ4K: true, weightQuant: 'q4k' })).toBe(true);
  });

  it('should accept non-fused with any weight type', () => {
    expect(isValidQuantConfig({ fusedQ4K: false, weightQuant: 'f16' })).toBe(true);
    expect(isValidQuantConfig({ fusedQ4K: false, weightQuant: 'q4k' })).toBe(true);
  });

  // Parametrized tests
  describe('Quant config matrix', () => {
    const configs = crossProduct(QUANT_PATH_DIMS);

    for (const config of configs) {
      const isValid = isValidQuantConfig(config);
      it(`[${configName(config)}] => ${isValid ? 'valid' : 'invalid'}`, () => {
        expect(typeof isValid).toBe('boolean');

        // Verify constraint
        if (config.fusedQ4K && config.weightQuant !== 'q4k') {
          expect(isValid).toBe(false);
        }
      });
    }
  });
});

describe('Full Quantization Path', () => {
  
  function traceQuantPath(config) {
    const {
      weightQuant,
      embeddingQuant,
      kvDtype,
      activationDtype = 'f32',
      hasSubgroups = true,
      hasF16 = true,
      preferFused = true,
    } = config;

    const matmulPath = selectMatmulPath({ weightQuant, activationDtype, hasSubgroups, hasF16, preferFused });

    // Dequant is only used when:
    // 1. Weight is quantized (not f32/f16)
    // 2. Matmul path is dequant_then_matmul (not fused)
    const usesDequant = matmulPath === 'dequant_then_matmul';

    return {
      embedding: selectEmbeddingPath({ embeddingQuant, hasF16 }),
      matmul: matmulPath,
      dequant: usesDequant
        ? selectDequantVariant({ quantType: weightQuant, hasSubgroups, hasF16 })
        : null,
      kv: selectKVDtype({ kvDtype, hasF16 }),
    };
  }

  it('should trace consistent F16 path', () => {
    const path = traceQuantPath({
      weightQuant: 'f16',
      embeddingQuant: 'f16',
      kvDtype: 'f16',
      activationDtype: 'f16', // Pure F16 path
      hasF16: true,
    });

    expect(path.embedding).toBe('gather_f16');
    expect(path.matmul).toBe('f16');
    expect(path.dequant).toBeNull();
    expect(path.kv).toBe('f16');
  });

  it('should trace Q4K fused path', () => {
    const path = traceQuantPath({
      weightQuant: 'q4k',
      embeddingQuant: 'f32',
      kvDtype: 'f32',
      hasSubgroups: true,
    });

    expect(path.matmul).toBe('q4_fused');
    expect(path.dequant).toBeNull(); // Fused path, no separate dequant
  });

  it('should trace Q4K dequant path when no subgroups', () => {
    const path = traceQuantPath({
      weightQuant: 'q4k',
      embeddingQuant: 'f32',
      kvDtype: 'f32',
      hasSubgroups: false,
    });

    expect(path.matmul).toBe('dequant_then_matmul');
    expect(path.dequant).toBe('shared_vec4');
  });
});
