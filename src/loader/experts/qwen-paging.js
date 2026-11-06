import { getDevice } from '../../gpu/device.js';
import { isWeightBuffer, isGpuBufferInstance } from '../../gpu/weight-buffer.js';
import { isBufferActive, releaseBuffer } from '../../memory/buffer-pool.js';
import { validateQwen35PagerConfig } from '../../config/schema/loading.schema.js';

/** @type {typeof import('./qwen-paging.js').prepareQwenExpertLayer} */
export async function prepareQwenExpertLayer(context, layer, consumerExperts, policy, recorder, signal) {
  const resolved = validateQwen35PagerConfig(policy);
  if (!resolved.enabled) return null;
  if (context.manifest?.moeConfig?.expertFormat !== 'qwen3_5') throw new Error('Qwen bank paging requires declared qwen3_5 experts.');
  const architecture = context.manifest.architecture;
  if (!architecture || typeof architecture !== 'object' || typeof architecture.numLayers !== 'number'
    || !Number.isSafeInteger(layer) || layer < 0 || layer >= architecture.numLayers) {
    throw new Error('Qwen bank paging requires a layer within the declared architecture.');
  }
  if (recorder) throw new Error('Qwen bank paging requires disableCommandBatching=true; recorded bank retirement is not supported.');
  signal?.throwIfAborted();
  const activeKey = `layer_${layer}_qwen35_packed`;
  const active = context.experts.get(activeKey);
  if (active) {
    // Map insertion order is the access order; no timestamps or global cache.
    context.experts.delete(activeKey);
    context.experts.set(activeKey, active);
  }
  const banks = [...context.experts.keys()].filter(key => /^layer_\d+_qwen35_packed$/.test(key));
  const permitted = resolved.maxResidentLayers - (active ? 0 : 1);
  const victims = banks.filter(key => key !== activeKey).slice(0, Math.max(0, banks.length - permitted));
  if (victims.length) {
    const device = getDevice();
    if (!device) throw new Error('Qwen bank retirement requires the owning GPU device.');
    // GPU commands retain every bank until execution completes, even on abort.
    await device.queue.onSubmittedWorkDone();
    signal?.throwIfAborted();
  }
  const evictedLayers = [];
  for (const key of victims) {
    const bank = context.experts.get(key);
    if (!bank) throw new Error(`Qwen bank ownership changed before retirement: ${key}.`);
    const index = Number(key.split('_')[1]);
    const prefix = `layer_${index}_expert_`;
    const buffers = new Set();
    for (const field of ['gateUp', 'gate', 'up', 'down']) {
      const weight = bank[/** @type {'gateUp' | 'gate' | 'up' | 'down'} */ (field)];
      if (isWeightBuffer(weight)) {
        buffers.add(weight.buffer);
        for (const value of Object.values(weight.materializations ?? {})) buffers.add(value.buffer);
      } else if (isGpuBufferInstance(weight)) buffers.add(weight);
    }
    if (!buffers.size) throw new Error(`Qwen bank ${key} has no materialized buffers.`);
    context.experts.delete(key);
    for (const alias of context.experts.keys()) if (alias.startsWith(prefix)) context.experts.delete(alias);
    for (const alias of consumerExperts.keys()) if (alias.startsWith(prefix)) consumerExperts.delete(alias);
    const errors = [];
    for (const buffer of buffers) {
      try {
        context.gpuBuffers.delete(buffer);
        if (isBufferActive(buffer)) releaseBuffer(buffer);
        else buffer.destroy();
      } catch (error) {
        context.gpuBuffers.add(buffer);
        errors.push(error);
      }
    }
    if (errors.length) throw new AggregateError(errors, `Qwen bank ${key} retirement failed; remaining buffers retain their loader owner.`);
    evictedLayers.push(index);
  }
  return { maxResidentLayers: resolved.maxResidentLayers,
    residentLayers: [...context.experts.keys()].filter(key => /^layer_\d+_qwen35_packed$/.test(key)).map(key => Number(key.split('_')[1])),
    evictedLayers };
}
