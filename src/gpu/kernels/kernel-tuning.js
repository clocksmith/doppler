

import { getKernelCapabilities } from '../device.js';
import { getKernelTuner, getTunerConfig } from '../kernel-tuner.js';
import { KERNEL_CONFIGS } from './kernel-configs.js';
import { createPipeline } from './pipeline-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { log } from '../../debug/index.js';

// ============================================================================
// Workgroup Size Tuning
// ============================================================================


export async function getTunedWorkgroupSize(
  operation,
  inputSizes = {}
) {
  try {
    const tuner = await getKernelTuner();
    const result = tuner.getCachedResult(operation, inputSizes);

    if (result) {
      return result.optimalWorkgroupSize;
    }

    // Run tuning if not cached
    const tuneResult = await tuner.tuneKernel(operation, inputSizes);
    return tuneResult.optimalWorkgroupSize;
  } catch (e) {
    log.warn('KernelTuning', `Tuning failed for ${operation}, using defaults: ${e.message}`);
    const { fallbackWorkgroupSizes } = getTunerConfig();
    const fallback = fallbackWorkgroupSizes?.[operation] ?? fallbackWorkgroupSizes?.default;
    if (!fallback) {
      throw new Error(`KernelTuning: missing fallback workgroup size for "${operation}".`);
    }
    return fallback;
  }
}

// ============================================================================
// Auto-Tuning
// ============================================================================


export async function autoTuneKernels(
  modelConfig = {}
) {
  const {
    hiddenSize = 4096,
    intermediateSize = 14336,
    numHeads = 32,
    headDim = 128,
    maxSeqLen = 4096,
    vocabSize = 32000,
  } = modelConfig;

  const tuner = await getKernelTuner();
  
  const results = {};

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


export async function prewarmKernels(
  options = {}
) {
  const caps = getKernelCapabilities();
  const mode = options.mode ?? 'parallel';
  const entries = Object.entries(KERNEL_CONFIGS)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([operation, variants]) => [
      operation,
      Object.entries(variants).sort(([a], [b]) => a.localeCompare(b))
    ]);

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
        } catch (e) {
          log.warn('KernelTuning', `Prewarm failed for ${operation}/${variant}: ${e.message}`);
        }
      }
    }
    log.debug('KernelTuning', `Prewarmed ${count} kernel pipelines`);
    return;
  }

  
  const jobs = [];
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
