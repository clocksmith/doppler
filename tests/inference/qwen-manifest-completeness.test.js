// Qwen 3.5 hybrid (linear + full attention) conversion configs and their
// converted manifests. Assertions check structure and cross-file derivation:
// the execution graph resolves through the real execution-v1 expander, kernel
// identities resolve to shipped WGSL entry points, dtype declarations agree,
// and local manifests carry what the conversion config and tensors imply.
// Tuning values (decode loop sizes, kernel choices) stay owned by the configs.
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

import { validateRequiredInferenceFields } from '../../src/inference/pipelines/text/config.js';
import { inferLinearNormMode } from '../../src/inference/pipelines/text/linear-attention.js';
import { expandExecutionV1 } from '../../src/config/schema/execution-v1.schema.js';
import { resolveMaterializedManifestFromConversionConfig } from '../../src/tooling/conversion-config-materializer.js';

const QWEN_MODEL_IDS = ['qwen-3-5-0-8b-q4k-ehaf16', 'qwen-3-5-2b-q4k-ehaf16'];
const QWEN_LAYER_TYPES = new Set(['linear_attention', 'full_attention']);

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

const kernelSourceCache = new Map();
function readKernelSource(kernelFile) {
  if (!kernelSourceCache.has(kernelFile)) {
    const path = `src/gpu/kernels/${kernelFile}`;
    assert.ok(existsSync(path), `kernel file ${kernelFile} must exist under src/gpu/kernels`);
    kernelSourceCache.set(kernelFile, readFileSync(path, 'utf8'));
  }
  return kernelSourceCache.get(kernelFile);
}

function assertExecutionGraphResolves(execution, label) {
  assert.equal(execution?.inlineKernelPath, true, `${label}: execution-v1 must inline the kernel path`);
  const steps = expandExecutionV1(execution);
  assert.ok(steps.some((step) => step.phase === 'decode'), `${label}: decode steps must be declared`);
  assert.ok(steps.some((step) => step.phase === 'prefill'), `${label}: prefill steps must be declared`);
  for (const step of steps) {
    const source = readKernelSource(step.kernel);
    assert.match(
      source,
      new RegExp(`@compute[^;{]*\\bfn\\s+${step.entry}\\s*\\(`),
      `${label}: ${step.kernel} must declare compute entry ${step.entry}`
    );
  }
  for (const [key, decl] of Object.entries(execution.kernels)) {
    if (decl.kernel.startsWith('attention_') && decl.kernel.includes('f16kv')) {
      assert.equal(decl.precision?.kvDtype, 'f16', `${label}: ${key} uses an f16kv kernel and must declare kvDtype f16`);
    }
  }
  return steps;
}

function kernelIdentity(kernels) {
  return Object.fromEntries(Object.entries(kernels).map(([key, decl]) => [
    key,
    { kernel: decl.kernel, entry: decl.entry, constants: decl.constants ?? null, precision: decl.precision ?? null },
  ]));
}

function assertQwenConversionConfig(config) {
  const label = config.output?.modelBaseId;
  assert.ok(QWEN_MODEL_IDS.includes(label), `unexpected qwen config ${label}`);
  assert.doesNotThrow(
    () => validateRequiredInferenceFields(config.inference, label),
    `${label}: conversion config must satisfy required inference-field validation`
  );
  assert.equal(config.inference.defaultKernelPath, undefined, `${label}: execution-v1 configs must not author defaultKernelPath`);

  const layerPattern = config.inference.layerPattern;
  assert.equal(layerPattern.type, 'custom', `${label}: hybrid attention requires an explicit layer list`);
  assert.ok(layerPattern.layerTypes.length > 0);
  for (const type of layerPattern.layerTypes) {
    assert.ok(QWEN_LAYER_TYPES.has(type), `${label}: unknown layer type ${type}`);
  }
  assert.ok(layerPattern.layerTypes.includes('linear_attention'), `${label}: hybrid model must contain linear-attention layers`);
  assert.ok(layerPattern.layerTypes.includes('full_attention'), `${label}: hybrid model must contain full-attention layers`);

  const steps = assertExecutionGraphResolves(config.execution, label);

  // The KV cache dtype must agree with the attention kernels the graph selects.
  const attentionKernels = steps.filter((step) => step.op === 'attention').map((step) => step.kernel);
  assert.ok(attentionKernels.length > 0, `${label}: graph must contain attention steps`);
  if (attentionKernels.every((kernel) => kernel.includes('f16kv'))) {
    assert.equal(config.session?.kvcache?.kvDtype, 'f16', `${label}: f16kv attention requires an f16 KV cache`);
  }

  // A quantized LM head must be executed by a quantized-weight kernel.
  const lmHead = steps.find((step) => step.op === 'lm_head');
  assert.ok(lmHead, `${label}: postLayer must contain lm_head`);
  if (config.quantization?.lmHead === 'q4k') {
    assert.match(lmHead.kernel, /q4/, `${label}: q4k lm_head must use a q4 kernel, got ${lmHead.kernel}`);
  }
}

const convConfigs = QWEN_MODEL_IDS.map((modelId) => readJson(`src/config/conversion/qwen3/${modelId}.json`));
for (const config of convConfigs) {
  assertQwenConversionConfig(config);
}

// Malformed graphs are rejected by the same expander, so the checks above are meaningful.
{
  const broken = structuredClone(convConfigs[0].execution);
  broken.decode = [...broken.decode, ['extra', 'not_a_declared_kernel']];
  assert.throws(() => expandExecutionV1(broken), /not_a_declared_kernel" not found in execution\.kernels/);
}

const localManifests = QWEN_MODEL_IDS
  .map((modelId, index) => ({ path: `models/local/${modelId}/manifest.json`, config: convConfigs[index] }))
  .filter(({ path }) => existsSync(path))
  .map(({ path, config }) => ({ manifest: readJson(path), config }));

if (localManifests.length === 0) {
  console.log('qwen-manifest-completeness.test: ok (conversion-config contract only; local qwen manifests unavailable)');
  process.exit(0);
}

for (const { manifest, config } of localManifests) {
  const label = manifest.modelId;
  const inference = manifest.inference;
  const arch = manifest.architecture;

  assert.equal(inference.schema, 'doppler.execution/v1');
  assert.equal(inference.defaultKernelPath, undefined, `${label}: manifests must not carry defaultKernelPath`);
  assert.doesNotThrow(() => validateRequiredInferenceFields(inference, label));
  assertExecutionGraphResolves(inference.execution, `${label} manifest`);

  // Manifest content is derived from the conversion config. Kernel digests are
  // checked by kernels:conversion-digests:check, not here.
  const materialized = resolveMaterializedManifestFromConversionConfig(config, manifest).inference;
  assert.deepEqual(kernelIdentity(inference.execution.kernels), kernelIdentity(materialized.execution.kernels), `${label}: kernel identities`);
  for (const section of ['preLayer', 'decode', 'prefill', 'postLayer']) {
    assert.deepEqual(inference.execution[section], materialized.execution[section], `${label}: execution.${section}`);
  }
  for (const field of ['layerPattern', 'rope', 'normalization', 'attention', 'ffn', 'output']) {
    assert.deepEqual(inference[field], materialized[field], `${label}: inference.${field}`);
  }
  for (const field of ['decodeLoop', 'perLayerInputs']) {
    assert.deepEqual(inference.session?.[field], config.session?.[field], `${label}: session.${field}`);
  }
  assert.deepEqual(inference.session?.compute?.defaults, config.session?.compute?.defaults, `${label}: compute defaults`);
  assert.equal(inference.session?.kvcache?.kvDtype, config.session?.kvcache?.kvDtype, `${label}: kv dtype`);
  assert.equal(inference.session?.execution, undefined, `${label}: session must not duplicate the execution graph`);
  if (config.quantization?.lmHead) {
    assert.equal(manifest.quantizationInfo?.lmHead, config.quantization.lmHead, `${label}: lm_head quantization`);
  }

  // Layer pattern covers every layer.
  const layerTypes = inference.layerPattern.layerTypes;
  assert.equal(layerTypes.length, arch.numLayers, `${label}: layerTypes must cover numLayers`);

  // mRoPE sections cover exactly the rotary half-dimension.
  const rope = inference.rope;
  if (Array.isArray(rope.mropeSection)) {
    const sectionSum = rope.mropeSection.reduce((sum, value) => sum + value, 0);
    assert.equal(sectionSum * 2, arch.headDim * rope.partialRotaryFactor, `${label}: mRoPE sections must span the rotary dim`);
  }

  // EOS ids are valid vocabulary ids.
  const eosIds = [].concat(manifest.eos_token_id);
  assert.ok(eosIds.length > 0, `${label}: eos_token_id must be declared`);
  for (const id of eosIds) {
    assert.ok(Number.isInteger(id) && id >= 0 && id < arch.vocabSize, `${label}: eos id ${id} outside vocab`);
  }

  // Tensors match each layer's attention type, and the declared linear norm
  // mode is what the runtime infers from the stored norm tensor.
  for (const field of ['linearNumKeyHeads', 'linearNumValueHeads', 'linearKeyHeadDim', 'linearValueHeadDim', 'linearConvKernelDim']) {
    assert.ok(Number.isInteger(arch[field]) && arch[field] > 0, `${label}: architecture.${field} must be a positive integer`);
  }
  const projectionLayout = {
    headVDim: arch.linearValueHeadDim,
    valueDim: arch.linearNumValueHeads * arch.linearValueHeadDim,
  };
  const has = (name) => Object.hasOwn(manifest.tensors ?? {}, name);
  assert.ok(
    has('model.language_model.lm_head.weight') || has('lm_head.weight') || inference.output?.tieWordEmbeddings === true,
    `${label}: lm_head must be materialized or tied`
  );
  for (const [layerIdx, layerType] of layerTypes.entries()) {
    const prefix = `model.language_model.layers.${layerIdx}`;
    if (layerType === 'linear_attention') {
      for (const name of ['in_proj_z', 'in_proj_a', 'in_proj_b', 'norm']) {
        assert.ok(has(`${prefix}.linear_attn.${name}.weight`), `${label}: layer ${layerIdx} missing linear_attn.${name}`);
      }
      assert.equal(has(`${prefix}.self_attn.q_norm.weight`), false, `${label}: layer ${layerIdx} linear layer has q_norm`);
      if (arch.linearNormMode != null) {
        assert.equal(
          inferLinearNormMode(manifest.tensors[`${prefix}.linear_attn.norm.weight`], projectionLayout),
          arch.linearNormMode,
          `${label}: layer ${layerIdx} norm tensor disagrees with architecture.linearNormMode`
        );
      }
    } else {
      assert.ok(has(`${prefix}.self_attn.q_norm.weight`), `${label}: layer ${layerIdx} missing q_norm`);
      assert.ok(has(`${prefix}.self_attn.k_norm.weight`), `${label}: layer ${layerIdx} missing k_norm`);
    }
  }
}

console.log('qwen-manifest-completeness.test: ok');
