import { log } from '../debug/index.js';

/**
 * Wraps a LoRA merge operation to include the adapter path in any error.
 *
 * @param {object} options
 * @param {Function} options.merge - The merge function to invoke.
 * @param {string} options.adapterPath - Path to the LoRA adapter being merged.
 * @param {string} [options.label] - Optional label for logging.
 * @returns {Promise<*>} The merge result.
 */
export async function runLoRAMerge(options = {}) {
  const { merge, adapterPath, label } = options;

  if (typeof merge !== 'function') {
    throw new Error('lora-runner: options.merge must be a function.');
  }
  if (!adapterPath || typeof adapterPath !== 'string') {
    throw new Error('lora-runner: options.adapterPath is required.');
  }

  const runLabel = label || 'lora-merge';
  log.debug('lora-runner', `Starting ${runLabel} with adapter: ${adapterPath}`);

  try {
    const result = await merge();
    log.debug('lora-runner', `${runLabel} complete for adapter: ${adapterPath}`);
    return result;
  } catch (error) {
    const enriched = new Error(
      `lora-runner: merge failed for adapter "${adapterPath}": ${error.message}`
    );
    enriched.cause = error;
    enriched.adapterPath = adapterPath;
    throw enriched;
  }
}
