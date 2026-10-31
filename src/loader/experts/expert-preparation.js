import { validateExpertPreparationConfig } from '../../config/schema/loading.schema.js';
import { supportsResidentExpertPreparation } from './expert-loader.js';
import { getDevice } from '../../gpu/device.js';

/** @type {import('./expert-preparation.js').createExpertPreparation} */
export function createExpertPreparation(loader, policy) {
  const { mode } = validateExpertPreparationConfig(policy);
  const format = loader.manifest?.moeConfig?.expertFormat;
  if (mode === 'eager-resident' && !supportsResidentExpertPreparation(format)) {
    throw new Error(`Eager resident expert preparation is unsupported for expertFormat=${String(format)}. `
      + 'Use on-demand loading for per-expert or layer-paged formats.');
  }
  const numExperts = loader.manifest?.moeConfig?.numExperts;
  if (typeof numExperts !== 'number' || !Number.isSafeInteger(numExperts) || numExperts < 1) {
    throw new Error('Expert preparation requires manifest.moeConfig.numExperts to be a positive integer.');
  }
  /** @type {import('./expert-preparation.js').ExpertPreparationReceipt} */
  const receipt = { mode, status: mode === 'on-demand' ? 'deferred' : 'pending',
    layerCount: null, preparedBanks: 0, expertsPerBank: numExperts };
  if (!loader.loadTiming) throw new Error('Expert preparation requires an active loader timing receipt.');
  loader.loadTiming.expertPreparation = receipt;

  return async (layerRange, { signal, onProgress }) => {
    signal?.throwIfAborted();
    receipt.layerCount = layerRange[1] - layerRange[0] + 1;
    if (mode === 'on-demand') return;
    receipt.status = 'running';
    const device = getDevice();
    try {
      for (let layer = layerRange[0]; layer <= layerRange[1]; layer++) {
        signal?.throwIfAborted();
        // Await the materialization owner; cancellation never races cleanup against it.
        const weights = await loader.loadExpert(layer, 0);
        if (weights.numExperts !== numExperts) {
          throw new Error(`Prepared expert bank at layer ${layer} disagrees with manifest.moeConfig.numExperts.`);
        }
        signal?.throwIfAborted();
        loader._assertResidentBudget(`expert bank ${layer}`);
        receipt.preparedBanks++;
        onProgress?.({ stage: 'experts', progress: 0.90 + 0.09 * receipt.preparedBanks / receipt.layerCount,
          layer: receipt.preparedBanks, total: receipt.layerCount,
          message: `Prepared expert bank for layer ${layer}` });
      }
      await device?.queue.onSubmittedWorkDone();
      signal?.throwIfAborted();
      receipt.status = 'complete';
    } catch (error) {
      receipt.status = signal?.aborted ? 'cancelled' : 'failed';
      // Submitted uploads remain owned until the queue settles, including aborts.
      try { await device?.queue.onSubmittedWorkDone(); }
      catch (settlement) {
        throw new AggregateError([error, settlement], 'Expert preparation and queue settlement failed.', { cause: error });
      }
      throw error;
    }
  };
}
