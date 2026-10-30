import { Qwen35Engine, prefillMathFeatures } from '/vendor/engine/qwen35.js';
import { roomQwen35Options } from '/vendor/engine/preset.js';
import { makeTokenizer, argmax } from '/vendor/engine/engine.js';
import { parseGGUFHeader } from '/vendor/engine/gguf.js';
import { loadCanonicalWeights } from './pooled-canonical.js';
import { observePooledUploads, verifyPooledGpu } from './verify-pooled-gpu.js';
import { encodeSharedPrompt } from './shared-input.js';
import { captureSingleTokenLayers } from './capture-pooled-boundaries.js';
import { measurePooledRequests } from './timing.js';

try {
  const config = await (await fetch('/probe-config.json')).json();
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter || adapter.info.vendor.toLowerCase() !== config.requiredVendor) throw new Error('Required physical adapter unavailable');
  const device = await adapter.requestDevice({ requiredFeatures: prefillMathFeatures(adapter, 'f32'), requiredLimits: {
    maxBufferSize: adapter.limits.maxBufferSize, maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize } });
  const gpuErrors = [];
  device.addEventListener('uncapturederror', event => gpuErrors.push(event.error.message));
  const headerResponse = await fetch('/model.gguf', { headers: { Range: 'bytes=0-67108863' } });
  if (headerResponse.status !== 206) throw new Error('GGUF metadata range failed');
  const meta = parseGGUFHeader(await headerResponse.arrayBuffer()).meta;
  const hf = await (await fetch('/canonical/config.json')).json();
  const text = hf.text_config;
  const weights = await loadCanonicalWeights('/canonical', text.layer_types);
  const nativeOptions = config.pooledNativeOptions ?? {};
  if (Object.keys(nativeOptions).some(key => key !== 'attnFlash') ||
      (nativeOptions.attnFlash !== undefined && typeof nativeOptions.attnFlash !== 'boolean')) throw new Error('Unsupported native Pooled options');
  const observation = observePooledUploads(Qwen35Engine, weights);
  let engine;
  try {
    engine = await Qwen35Engine.create({ device, meta, weights, layerRange: [0, text.num_hidden_layers],
      hasEmbed: true, hasHead: true, maxSeq: config.contract.shared.maxSeqLen, vocab: text.vocab_size,
      ...roomQwen35Options(new URLSearchParams()), ...nativeOptions, prefillMath: 'f32', adapterInfo: adapter.info });
  } finally { observation.restore(); }
  if (!engine.gpuSample || engine.kvQ8 || engine.mtp) throw new Error('Native execution settings differ from common contract');
  const kvDtype = engine.kvQ8 ? 'q8' : engine.flash ? 'f16' : 'f32';
  if (kvDtype !== config.contract.shared.kvDtype) throw new Error('Native Pooled KV dtype differs from common contract');
  const materialization = await verifyPooledGpu(device, weights, observation.uploaded, config.contract.materializationObservation);
  const tokenizer = makeTokenizer(await (await fetch('/canonical/tokenizer.json')).json());
  const promptIds = encodeSharedPrompt(tokenizer, config.prompt, config.contract.promptFormat);
  if (config.timing) {
    const samples = await measurePooledRequests(engine, device, promptIds, config.contract, config.timing);
    if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
    globalThis.RESULT = { schema: 'pooled.equal-weight-timing-pilot.v1', diagnosticOnly: true,
      samples, promptIds, materialization, input: weights.canonicalProof,
      adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device },
      settings: { kvDtype, attnFlash: engine.flash, maxSeq: engine.maxSeq }, gpuErrors,
      claim: 'Unpaired resident pilot only; not claim-grade performance evidence.' };
  } else {
  engine.reset();
  await engine.prefillTokens(promptIds.slice(0, -1));
  const logits = Float32Array.from(await engine.forwardToken(promptIds.at(-1)));
  const tokenIds = [argmax(logits)];
  for (let i = 1; i < config.probeTokens && !config.contract.shared.eosTokenIds.includes(tokenIds.at(-1)); i++) tokenIds.push((await engine.forwardTokenIds(tokenIds.at(-1))).ids[0]);
  const logitRows = [];
  if (config.captureFullSequenceLogits) {
    engine.reset(); await engine.prefillTokens(promptIds.slice(0, -1));
    let current = Float32Array.from(await engine.forwardToken(promptIds.at(-1)));
    const materializedTokenIds = [];
    for (let index = 0; index < tokenIds.length; index++) {
      const next = argmax(current); materializedTokenIds.push(next);
      if (config.contract.gates.logitPositions.includes(index)) logitRows.push({ index, tokenId: next, values: Array.from(current) });
      if (next !== tokenIds[index]) throw new Error(`Native GPU and materialized-head tokens differ at ${index}`);
      if (index + 1 < tokenIds.length) current = Float32Array.from(await engine.forwardToken(next));
    }
  }
  engine.reset();
  const single = await captureSingleTokenLayers(engine, promptIds[0]);
  const singleTokenLogits = single.logits;
  engine.reset();
  let sequential;
  for (const id of promptIds) sequential = await engine.forwardToken(id);
  const sequentialPromptLogits = Array.from(sequential);
  if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
  globalThis.RESULT = { schema: 'pooled.equal-weight-untimed-probe.v1', input: weights.canonicalProof, materialization,
    engineVariant: config.pooledRoundingPreparation ? 'pooled-snapshot-explicit-rne' : 'pooled-native-f32',
    settings: { kvDtype, attnFlash: engine.flash, kvBufferBytes: engine.kvBufBytes, maxSeq: engine.maxSeq },
    promptIds, tokenIds, stopReason: config.contract.shared.eosTokenIds.includes(tokenIds.at(-1)) ? 'eos' : 'max-tokens', logitRows,
    text: tokenizer.decode(tokenIds), firstLogits: Array.from(logits), singleTokenLogits, singleTokenLayers: single.layers, sequentialPromptLogits, gpuErrors,
    adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device },
    claim: 'Untimed canonical F32 probe. Cross-engine parity remains a separate gate.' };
  }
} catch (error) { globalThis.RESULT = { error: String(error.stack || error), completedSamples: error.completedSamples ?? [] }; console.error(error); }
