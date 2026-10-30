import { Qwen35Engine, prefillMathFeatures } from '/vendor/engine/qwen35.js';
import { roomQwen35Options } from '/vendor/engine/preset.js';
import { makeTokenizer } from '/vendor/engine/engine.js';
import { parseGGUFHeader } from '/vendor/engine/gguf.js';
import { loadCanonicalWeights } from './pooled-canonical.js';
import { observePooledUploads } from './verify-pooled-gpu.js';

// Initialization is shared with the untimed probe. No materialization readback
// or generation policy runs here; their existing owners remain explicit callers.
export async function initializePooledSession(config, { observeUploads = false } = {}) {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.vendor.toLowerCase() !== config.requiredVendor) throw new Error('Required physical adapter unavailable');
  const device = await adapter.requestDevice({ requiredFeatures: prefillMathFeatures(adapter, 'f32'), requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } });
  const gpuErrors = [];
  device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  let engine;
  try {
    const response = await fetch('/model.gguf', { headers: { Range: 'bytes=0-67108863' } });
    if (response.status !== 206) throw new Error('GGUF metadata range failed');
    const meta = parseGGUFHeader(await response.arrayBuffer()).meta;
    const hf = await (await fetch('/canonical/config.json')).json();
    const text = hf.text_config;
    const weights = await loadCanonicalWeights('/canonical', text.layer_types);
    const nativeOptions = config.pooledNativeOptions ?? {};
    if (Object.keys(nativeOptions).some(key => key !== 'attnFlash') ||
        (nativeOptions.attnFlash !== undefined && typeof nativeOptions.attnFlash !== 'boolean')) throw new Error('Unsupported native Pooled options');
    const observation = observeUploads ? observePooledUploads(Qwen35Engine, weights) : null;
    try {
      engine = await Qwen35Engine.create({ device, meta, weights, layerRange: [0, text.num_hidden_layers],
        hasEmbed: true, hasHead: true, maxSeq: config.contract.shared.maxSeqLen, vocab: text.vocab_size,
        ...roomQwen35Options(new URLSearchParams()), ...nativeOptions, prefillMath: 'f32', adapterInfo: adapter.info });
    } finally { observation?.restore(); }
    if (!engine.gpuSample || engine.kvQ8 || engine.mtp) throw new Error('Native execution settings differ from common contract');
    const kvDtype = engine.kvQ8 ? 'q8' : engine.flash ? 'f16' : 'f32';
    if (kvDtype !== config.contract.shared.kvDtype) throw new Error('Native Pooled KV dtype differs from common contract');
    const tokenizer = makeTokenizer(await (await fetch('/canonical/tokenizer.json')).json());
    return { adapter, device, engine, weights, observation, tokenizer, gpuErrors, kvDtype };
  } catch (error) {
    engine?.destroy?.();
    device.destroy();
    throw error;
  }
}
