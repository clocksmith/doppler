/**
 * LoRA Loader - Loads LoRA weights from an adapter manifest.
 *
 * @module loader/lora-loader
 */

import { parseLoRATensorName, toFloat32 } from './lora-utils.js';
import { buildTensorLocations } from './shard-resolver.js';

/**
 * Load LoRA weights from an adapter manifest.
 *
 * @param {import('../storage/rdrr-format.js').RDRRManifest} manifest
 * @param {(name: string, toGPU: boolean, silent: boolean) => Promise<GPUBuffer | import('../gpu/weight-buffer.js').WeightBuffer | import('../gpu/weight-buffer.js').CpuWeightBuffer | Float32Array | Uint8Array | null>} loadTensor
 * @returns {Promise<import('../inference/pipeline/lora.js').LoRAAdapter>}
 */
export async function loadLoRAWeights(manifest, loadTensor) {
  const isLoRA = manifest.adapterType === 'lora' || manifest.modelType === 'lora' || !!manifest.loraConfig;
  if (!isLoRA) {
    throw new Error('Manifest is not a LoRA adapter');
  }
  if (!manifest.loraConfig) {
    throw new Error('LoRA manifest missing loraConfig');
  }

  // Build tensor locations for this manifest (local scope)
  const tensorLocations = await buildTensorLocations(manifest, {
    // LoRA adapters typically use external tensors.json or inline
    hasCustomLoader: false, // Assuming standard loading for now, or pass via context if needed
  });

  /** @type {import('../inference/pipeline/lora.js').LoRAAdapter} */
  const adapter = {
    name: manifest.modelId,
    version: typeof manifest.version === 'string' ? manifest.version : String(manifest.version),
    baseModel: manifest.baseModel,
    rank: manifest.loraConfig.rank,
    alpha: manifest.loraConfig.alpha,
    targetModules: /** @type {import('../inference/pipeline/lora.js').LoRAModuleName[] | undefined} */ (manifest.loraConfig.targetModules),
    layers: new Map(),
  };

  // Helper to load from local map
  // We need to inject the locations into the loader or pass them.
  // The passed `loadTensor` usually uses the loader's `tensorLocations`.
  // But `DopplerLoader` swaps `this.manifest` and `this._buildTensorLocations()` temporarily.
  // Here we should probably pass the location directly to a lower level loader, OR follow the swap pattern.
  // Given we want to decouple, `loadTensor` callback should ideally accept a location or we simulate the swap in caller.

  // IF we follow the pattern where caller swaps context:
  // return loadTensor(name, false, true);

  // However, `DopplerLoader.loadLoRAWeights` implementation:
  // 1. Swaps manifest.
  // 2. Calls `_buildTensorLocations` (updates `this.tensorLocations`).
  // 3. Iterates `this.tensorLocations`.
  // 4. Calls `_loadTensor` (uses `this.tensorLocations`).

  // To extract this PURELY, `loadLoRAWeights` here should probably take `tensorLocations` map populated by caller?
  // Or it should do the population itself and return the result, but it needs to call `loadTensor` which depends on `tensorLocations` being set in the loader?

  // If `loadTensor` relies on `this.tensorLocations` in `DopplerLoader`, then `DopplerLoader` MUST wrap this call with state swap.
  // So this function here is just the iteration logic.

  // Let's assume `loadTensor` works for the names we find in `tensorLocations`.

  for (const name of tensorLocations.keys()) {
    const parsed = parseLoRATensorName(name);
    if (!parsed) continue;

    // We expect the caller to have set up the environment such that `loadTensor` finds this name.
    const tensor = await loadTensor(name, false, true);
    if (!tensor) continue;
    const data = toFloat32(tensor);

    const layer = adapter.layers.get(parsed.layer) || {};
    const scale = adapter.rank > 0 ? adapter.alpha / adapter.rank : 1;
    if (!layer[parsed.module]) {
      layer[parsed.module] = {
        a: new Float32Array(0),
        b: new Float32Array(0),
        rank: adapter.rank,
        alpha: adapter.alpha,
        scale,
      };
    }

    if (parsed.kind === 'a') {
      layer[parsed.module].a = data;
    } else {
      layer[parsed.module].b = data;
    }

    adapter.layers.set(parsed.layer, layer);
  }

  return adapter;
}
