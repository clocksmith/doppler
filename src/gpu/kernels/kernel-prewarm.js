import { getKernelCapabilities } from '../device.js';
import { KERNEL_CONFIGS } from './kernel-configs.js';
import { createPipeline, clearPipelineCaches } from './pipeline-cache.js';
import { clearShaderCaches } from './shader-cache.js';
import { hasRequiredFeatures } from './feature-check.js';
import { log } from '../../debug/index.js';

export async function prewarmKernels(options = {}) {
  const capabilities = getKernelCapabilities();
  const mode = options.mode ?? 'parallel';
  const entries = Object.entries(KERNEL_CONFIGS)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([operation, variants]) => [
      operation,
      Object.entries(variants).sort(([left], [right]) => left.localeCompare(right)),
    ]);

  try {
    if (mode === 'sequential') {
      let count = 0;
      for (const [operation, variants] of entries) {
        for (const [variant, config] of variants) {
          if (config.requires && !hasRequiredFeatures(config.requires, capabilities)) {
            continue;
          }
          try {
            await createPipeline(operation, variant);
            count += 1;
          } catch (error) {
            log.warn('KernelPrewarm', `Prewarm failed for ${operation}/${variant}: ${error.message}`);
          }
        }
      }
      log.debug('KernelPrewarm', `Prewarmed ${count} kernel pipelines`);
      return;
    }

    const jobs = [];
    for (const [operation, variants] of entries) {
      for (const [variant, config] of variants) {
        if (config.requires && !hasRequiredFeatures(config.requires, capabilities)) {
          continue;
        }
        jobs.push(
          createPipeline(operation, variant)
            .then(() => {})
            .catch((error) => {
              log.warn('KernelPrewarm', `Prewarm failed for ${operation}/${variant}: ${error.message}`);
            })
        );
      }
    }
    await Promise.all(jobs);
    log.debug('KernelPrewarm', `Prewarmed ${jobs.length} kernel pipelines`);
  } catch (error) {
    clearPipelineCaches();
    clearShaderCaches();
    throw error;
  }
}
