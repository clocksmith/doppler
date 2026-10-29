import { Qwen35Engine, prefillMathFeatures } from '/vendor/engine/qwen35.js';
import { roomQwen35Options } from '/vendor/engine/preset.js';
import { makeTokenizer, argmax } from '/vendor/engine/engine.js';
import { parseGGUFHeader } from '/vendor/engine/gguf.js';
import { loadCanonicalWeights } from './pooled-canonical.js';
import { observePooledUploads, verifyPooledGpu } from './verify-pooled-gpu.js';
import { encodeSharedPrompt } from './shared-input.js';
import { captureSingleTokenLayers } from './capture-pooled-boundaries.js';

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
  const observation = observePooledUploads(Qwen35Engine, weights);
  let engine;
  try {
    engine = await Qwen35Engine.create({ device, meta, weights, layerRange: [0, text.num_hidden_layers],
      hasEmbed: true, hasHead: true, maxSeq: config.contract.shared.maxSeqLen, vocab: text.vocab_size,
      ...roomQwen35Options(new URLSearchParams()), prefillMath: 'f32', adapterInfo: adapter.info });
  } finally { observation.restore(); }
  if (!engine.gpuSample || engine.kvQ8 || engine.mtp) throw new Error('Native execution settings differ from common contract');
  const materialization = await verifyPooledGpu(device, weights, observation.uploaded, config.contract.materializationObservation);
  const tokenizer = makeTokenizer(await (await fetch('/canonical/tokenizer.json')).json());
  const promptIds = encodeSharedPrompt(tokenizer, config.prompt, config.contract.promptFormat);
  engine.reset();
  await engine.prefillTokens(promptIds.slice(0, -1));
  const logits = Float32Array.from(await engine.forwardToken(promptIds.at(-1)));
  const tokenIds = [argmax(logits)];
  for (let i = 1; i < config.probeTokens; i++) tokenIds.push((await engine.forwardTokenIds(tokenIds.at(-1))).ids[0]);
  engine.reset();
  const single = await captureSingleTokenLayers(engine, promptIds[0]);
  const singleTokenLogits = single.logits;
  engine.reset();
  let sequential;
  for (const id of promptIds) sequential = await engine.forwardToken(id);
  const sequentialPromptLogits = Array.from(sequential);
  if (gpuErrors.length) throw new Error(gpuErrors.join('\n'));
  globalThis.RESULT = { schema: 'pooled.equal-weight-untimed-probe.v1', input: weights.canonicalProof, materialization,
    promptIds, tokenIds, text: tokenizer.decode(tokenIds), firstLogits: Array.from(logits), singleTokenLogits, singleTokenLayers: single.layers, sequentialPromptLogits, gpuErrors,
    adapter: { vendor: adapter.info.vendor, architecture: adapter.info.architecture, device: adapter.info.device },
    claim: 'Untimed canonical F32 probe. Cross-engine parity remains a separate gate.' };
} catch (error) { globalThis.RESULT = { error: String(error.stack || error) }; console.error(error); }
