/**
 * Kernel Tuning - Auto-tuning and kernel prewarming
 *
 * Provides utilities for tuning kernel workgroup sizes and
 * prewarming kernel pipelines for optimal performance.
 *
 * @module gpu/kernels/kernel-tuning
 */

import { getKernelCapabilities } from '../device.js';
import { getKernelTuner } from '../kernel-tuner.js';
import { KERNEL_CONFIGS } from './kernel-configs.js';
import { createPipeline } from './pipeline-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { log } from '../../debug/index.js';

// ============================================================================
// Workgroup Size Tuning
// ============================================================================

/**
 * Get tuned workgroup size for an operation
 */
export async function getTunedWorkgroupSize(
  operation: string,
  inputSizes: Record<string, number> = {}
): Promise<[number, number, number]> {
  try {
    const tuner = await getKernelTuner();
    const result = tuner.getCachedResult(operation, inputSizes);

    if (result) {
      return result.optimalWorkgroupSize;
    }

    // Run tuning if not cached
    const tuneResult = await tuner.tuneKernel(operation, inputSizes);
    return tuneResult.optimalWorkgroupSize;
  } catch (e: any) {
    log.warn('KernelTuning', `Tuning failed for ${operation}, using defaults: ${e.message}`);
    // Return defaults based on operation
    switch (operation) {
      case 'matmul':
        return [16, 16, 1];
      case 'attention':
      case 'rmsnorm':
      case 'softmax':
        return [256, 1, 1];
      case 'dequant':
        return [64, 1, 1];
      default:
        return [256, 1, 1];
    }
  }
}

// ============================================================================
// Auto-Tuning
// ============================================================================

/**
 * Run auto-tuning for all kernels with given model config
 */
export async function autoTuneKernels(
  modelConfig: Record<string, number> = {}
): Promise<Record<string, any>> {
  const {
    hiddenSize = 4096,
    intermediateSize = 14336,
    numHeads = 32,
    headDim = 128,
    maxSeqLen = 4096,
    vocabSize = 32000,
  } = modelConfig;

  const tuner = await getKernelTuner();
  const results: Record<string, any> = {};

  // Tune matmul for common sizes
  results.matmul_hidden = await tuner.tuneKernel('matmul', {
    M: 1, N: hiddenSize, K: hiddenSize,
  });
  results.matmul_ffn = await tuner.tuneKernel('matmul', {
    M: 1, N: intermediateSize, K: hiddenSize,
  });

  // Tune attention
  results.attention = await tuner.tuneKernel('attention', {
    seqLen: 1, numHeads, headDim,
  });

  // Tune softmax (LM head output)
  results.softmax = await tuner.tuneKernel('softmax', {
    innerSize: vocabSize, outerSize: 1,
  });

  // Tune RMSNorm
  results.rmsnorm = await tuner.tuneKernel('rmsnorm', {
    hiddenSize, numTokens: 1,
  });

  // Tune dequant
  results.dequant = await tuner.tuneKernel('dequant', {
    numBlocks: 1000,
  });

  log.debug('KernelTuning', `Auto-tuning complete: ${JSON.stringify(results)}`);
  return results;
}

// ============================================================================
// Pipeline Prewarming
// ============================================================================

/**
 * Prewarm all supported kernel pipelines
 */
export async function prewarmKernels(
  options: { mode?: 'parallel' | 'sequential' } = {}
): Promise<void> {
  const caps = getKernelCapabilities();
  const mode = options.mode ?? 'parallel';
  const entries = Object.entries(KERNEL_CONFIGS)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([operation, variants]) => [
      operation,
      Object.entries(variants).sort(([a], [b]) => a.localeCompare(b))
    ] as const);

  if (mode === 'sequential') {
    let count = 0;
    for (const [operation, variants] of entries) {
      for (const [variant, cfg] of variants) {
        if (cfg.requires && !hasRequiredFeatures(cfg.requires, caps)) {
          continue;
        }
        try {
          await createPipeline(operation, variant);
          count += 1;
        } catch (e: any) {
          log.warn('KernelTuning', `Prewarm failed for ${operation}/${variant}: ${e.message}`);
        }
      }
    }
    log.debug('KernelTuning', `Prewarmed ${count} kernel pipelines`);
    return;
  }

  const jobs: Promise<void>[] = [];
  for (const [operation, variants] of entries) {
    for (const [variant, cfg] of variants) {
      if (cfg.requires && !hasRequiredFeatures(cfg.requires, caps)) {
        continue;
      }
      jobs.push(
        createPipeline(operation, variant)
          .then(() => {}) // Ignore the pipeline result
          .catch((e) => {
            log.warn('KernelTuning', `Prewarm failed for ${operation}/${variant}: ${e.message}`);
          })
      );
    }
  }

  await Promise.all(jobs);
  log.debug('KernelTuning', `Prewarmed ${jobs.length} kernel pipelines`);
}
