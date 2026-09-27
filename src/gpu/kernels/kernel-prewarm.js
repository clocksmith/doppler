import { getKernelCapabilities } from '../device.js';
import { getKernelRegistry } from './kernel-configs.js';
import { createPipeline, clearPipelineCaches } from './pipeline-cache.js';
import { clearShaderCaches } from './shader-cache.js';
import { getKernelWgslRequirements, hasRequiredFeatures } from './feature-check.js';
import { log } from '../../debug/index.js';

export function listPrewarmKernels(registry, capabilities) {
  return Object.entries(registry.configs)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([operation, variants]) => [
      operation,
      Object.entries(variants)
        .sort(([left], [right]) => left.localeCompare(right))
        .filter(([, config]) => hasRequiredFeatures(
          config.requires, capabilities, getKernelWgslRequirements(config)
        )),
    ]);
}

export async function prewarmKernels(options = {}) {
  const capabilities = getKernelCapabilities();
  const mode = options.mode ?? 'parallel';
  const registry = getKernelRegistry();
  const entries = listPrewarmKernels(registry, capabilities);

  try {
    if (mode === 'sequential') {
      let count = 0;
      for (const [operation, variants] of entries) {
        for (const [variant] of variants) {
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
      for (const [variant] of variants) {
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
