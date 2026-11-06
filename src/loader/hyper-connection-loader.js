import { createTensor } from '../gpu/tensor.js';
import { getWeightDtype, isWeightBuffer } from '../gpu/weight-buffer.js';
import { castF16ToF32 } from '../gpu/kernels/cast.js';
import {
  GATED_RESIDUAL_SUFFIXES, HYPER_CONNECTION_MIXER_PREFIXES, HYPER_CONNECTION_SUFFIXES,
  LAYER_PREFIXES, NGRAM_PLE_SUFFIXES, QSA_SUFFIXES,
} from './layer-tensor-aliases.js';

/** @param {string} message @returns {never} */
function fail(message) {
  throw new Error(`[HyperConnectionLoader] ${message}`);
}

/** @param {import('./hyper-connection-loader.js').MechanismLoaderContext} ctx
 * @param {readonly string[]} prefixes @param {readonly string[]} suffixes
 * @param {readonly number[]} shape @param {string | null} role @param {boolean} tensor */
function binding(ctx, prefixes, suffixes, shape, role, tensor) {
  const candidates = prefixes.flatMap(prefix => suffixes.map(suffix => `${prefix}.${suffix}`));
  const matches = candidates.filter(name => ctx.tensorLocations.has(name));
  if (matches.length !== 1) fail(`Expected one ${candidates[0]}; found ${matches.length}.`);
  const name = matches[0];
  const location = ctx.tensorLocations.get(name);
  if (!location) fail(`Missing location for ${name}.`);
  const logicalShape = location.originalShape ?? location.shape;
  if (logicalShape.length !== shape.length || logicalShape.some((value, index) => value !== shape[index])) {
    fail(`${name} requires shape [${shape}]; got [${logicalShape}].`);
  }
  return { name, location, shape, role, tensor };
}

/** Preflight the complete group before any of its weights are allocated.
 * @param {import('./hyper-connection-loader.js').MechanismLoaderContext} ctx
 * @param {Record<string, ReturnType<typeof binding>>} bindings */
async function materialize(ctx, bindings) {
  /** @type {Record<string, import('../gpu/tensor.js').Tensor | import('../gpu/weight-buffer.js').WeightBuffer>} */
  const result = {};
  for (const [key, spec] of Object.entries(bindings)) {
    const loaded = await ctx.loadTensor(spec.name, true, false, spec.role ?? undefined);
    if (!loaded) fail(`Required tensor ${spec.name} could not be loaded.`);
    if (loaded instanceof Float32Array || loaded instanceof Uint8Array) {
      fail(`${spec.name} requires GPU materialization.`);
    }
    const dtype = getWeightDtype(loaded);
    const buffer = isWeightBuffer(loaded) ? loaded.buffer : loaded;
    const elements = spec.shape.reduce((a, b) => a * b, 1);
    if (spec.tensor) {
      if ((dtype !== 'f16' && dtype !== 'f32') || buffer.size < elements * (dtype === 'f16' ? 2 : 4)) {
        fail(`${spec.name} requires a complete f16 or f32 vector.`);
      }
      const vector = createTensor(buffer, dtype, [elements], spec.name);
      if (dtype === 'f16') {
        if (!(ctx.gpuBuffers instanceof Set)) fail(`${spec.name} requires a GPU buffer owner before materialization.`);
        const converted = await castF16ToF32(vector);
        ctx.gpuBuffers.add(converted.buffer);
        result[key] = converted;
      } else result[key] = vector;
    } else {
      if (!isWeightBuffer(loaded) || (dtype !== 'f16' && dtype !== 'f32')) {
        fail(`${spec.name} requires a WeightBuffer with declared f16 or f32 projection storage and layout.`);
      }
      if (loaded.shape.length !== spec.shape.length || loaded.shape.some((value, index) => value !== spec.shape[index])
        || buffer.size < elements * (dtype === 'f16' ? 2 : 4)) {
        fail(`${spec.name} has an incomplete or mismatched projection representation.`);
      }
      result[key] = loaded;
    }
  }
  return result;
}

/** @param {Awaited<ReturnType<typeof materialize>>} weights @param {string} key */
function vectorWeight(weights, key) {
  const value = weights[key];
  if (!value || isWeightBuffer(value)) fail(`Expected the materialized vector ${key}.`);
  return value;
}

/** @param {Awaited<ReturnType<typeof materialize>>} weights @param {string} key */
function projectionWeight(weights, key) {
  const value = weights[key];
  if (!isWeightBuffer(value)) fail(`Expected the materialized projection ${key}.`);
  return value;
}

/** @param {import('./hyper-connection-loader.js').MechanismLoaderContext} ctx
 * @param {Record<string, ReturnType<typeof binding>>} bindings */
async function materializeGated(ctx, bindings) {
  const weights = await materialize(ctx, bindings);
  return { norm: vectorWeight(weights, 'norm'), down: projectionWeight(weights, 'down'),
    up: projectionWeight(weights, 'up'), inject: bindings.inject ? projectionWeight(weights, 'inject') : null };
}

/** @param {import('./hyper-connection-loader.js').MechanismLoaderContext} ctx
 * @param {readonly string[]} prefixes @param {Record<string, readonly string[]>} suffixes
 * @param {string} role @param {boolean} inject */
function gatedBindings(ctx, prefixes, suffixes, role, inject) {
  const hyper = ctx.hyperConnections;
  if (!hyper) fail('Gated residual weights require the resolved hyperConnections contract.');
  const { streamCount, streamWidth, projectionRank } = hyper;
  const width = streamCount * streamWidth;
  /** @type {Record<string, ReturnType<typeof binding>>} */
  const bindings = {
    norm: binding(ctx, prefixes, suffixes.norm, [width], null, true),
    down: binding(ctx, prefixes, suffixes.down, [projectionRank, width], `${role}_down`, false),
    up: binding(ctx, prefixes, suffixes.up, [width, projectionRank], `${role}_up`, false),
  };
  if (inject) bindings.inject = binding(ctx, prefixes, suffixes.inject, [streamCount, width], `${role}_inject`, false);
  return bindings;
}

/** @type {typeof import('./hyper-connection-loader.js').loadHyperConnectionLayerWeights} */
export async function loadHyperConnectionLayerWeights(ctx, layerIdx) {
  if (ctx.hyperConnections == null) return {};
  const prefixes = LAYER_PREFIXES(layerIdx);
  const attention = gatedBindings(ctx, prefixes, HYPER_CONNECTION_SUFFIXES.attention, 'attn_hc', true);
  const mlp = gatedBindings(ctx, prefixes, HYPER_CONNECTION_SUFFIXES.mlp, 'mlp_hc', true);
  const indexed = ctx.indexedAttention;
  const ngram = ctx.ngramEmbedding;
  const width = ctx.hyperConnections.streamCount * ctx.hyperConnections.streamWidth;
  const qsa = indexed?.layers.includes(layerIdx) ? {
    projection: binding(ctx, prefixes, QSA_SUFFIXES.projection,
      [(indexed.queryHeads + indexed.keyHeads) * indexed.headDim, ctx.hyperConnections.streamWidth], 'qsa_index_qk', false),
    qNorm: binding(ctx, prefixes, QSA_SUFFIXES.qNorm, [indexed.headDim], null, true),
    kNorm: binding(ctx, prefixes, QSA_SUFFIXES.kNorm, [indexed.headDim], null, true),
  } : null;
  const ple = ngram?.layers.some(layer => layer.decoderLayerIndex === layerIdx) ? {
    keyProj: binding(ctx, prefixes, NGRAM_PLE_SUFFIXES.keyProj, [width, ngram.embedDim], 'ple_key_proj', false),
    valueProj: binding(ctx, prefixes, NGRAM_PLE_SUFFIXES.valueProj, [ctx.hyperConnections.streamWidth, ngram.embedDim], 'ple_value_proj', false),
    normKey: binding(ctx, prefixes, NGRAM_PLE_SUFFIXES.normKey, [width], null, true),
    normQuery: binding(ctx, prefixes, NGRAM_PLE_SUFFIXES.normQuery, [width], null, true),
    normConv: binding(ctx, prefixes, NGRAM_PLE_SUFFIXES.normConv, [width], null, true),
    conv: binding(ctx, prefixes, NGRAM_PLE_SUFFIXES.conv, [width, 1, ngram.convKernelSize], null, true),
  } : null;
  const hyperConnections = { attention: await materializeGated(ctx, attention), mlp: await materializeGated(ctx, mlp) };
  const qsaWeights = qsa ? await materialize(ctx, qsa) : null;
  const pleWeights = ple ? await materialize(ctx, ple) : null;
  return { hyperConnections,
    qsaIndexer: qsaWeights ? { projection: projectionWeight(qsaWeights, 'projection'),
      qNorm: vectorWeight(qsaWeights, 'qNorm'), kNorm: vectorWeight(qsaWeights, 'kNorm') } : null,
    ple: pleWeights ? { keyProj: projectionWeight(pleWeights, 'keyProj'), valueProj: projectionWeight(pleWeights, 'valueProj'),
      normKey: vectorWeight(pleWeights, 'normKey'), normQuery: vectorWeight(pleWeights, 'normQuery'),
      normConv: vectorWeight(pleWeights, 'normConv'), conv: vectorWeight(pleWeights, 'conv') } : null };
}

/** @type {typeof import('./hyper-connection-loader.js').loadHyperConnectionMixerWeights} */
export async function loadHyperConnectionMixerWeights(ctx) {
  if (ctx.hyperConnections == null) return null;
  const suffixes = Object.fromEntries(Object.entries(GATED_RESIDUAL_SUFFIXES).map(([key, value]) => [key, [value]]));
  return materializeGated(ctx, gatedBindings(ctx, HYPER_CONNECTION_MIXER_PREFIXES, suffixes, 'hc_mixer', false));
}
