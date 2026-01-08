/**
 * Config-driven probe helpers for targeted GPU buffer inspection.
 *
 * Probes let us read specific token/dimension values without adding
 * ad-hoc logs in the code. Probes are configured via runtime.debug.probes.
 */

import { trace } from '../../debug/index.js';
import { getDevice } from '../../gpu/device.js';
import { allowReadback } from '../../gpu/perf-guards.js';

/** @type {Record<import('../../config/schema/index.js').ProbeStage, import('../../config/schema/index.js').TraceCategory>} */
const STAGE_DEFAULT_CATEGORY = {
  embed_out: 'embed',
  // Attention stages (per-layer)
  attn_input: 'attn',
  attn_normed: 'attn',
  q_proj: 'attn',
  k_proj: 'attn',
  v_proj: 'attn',
  q_rope: 'attn',
  k_rope: 'attn',
  attn_scores: 'attn',
  attn_out: 'attn',
  o_proj: 'attn',
  post_attn: 'attn',
  // FFN stages (per-layer)
  ffn_normed: 'ffn',
  ffn_in: 'ffn',
  ffn_gate: 'ffn',
  ffn_up: 'ffn',
  ffn_act: 'ffn',
  ffn_out: 'ffn',
  layer_out: 'ffn',
  // Final stages
  pre_final_norm: 'logits',
  final_norm: 'logits',
  logits: 'logits',
  logits_final: 'logits',
};

/**
 * @param {number[] | null | undefined} layers
 * @param {number | undefined} layerIdx
 * @returns {boolean}
 */
function matchesLayer(layers, layerIdx) {
  if (!layers || layers.length === 0) return true;
  if (layerIdx === undefined || layerIdx === null) return false;
  return layers.includes(layerIdx);
}

/**
 * @param {number[] | null | undefined} tokens
 * @param {number} numTokens
 * @returns {number[]}
 */
function resolveTokens(tokens, numTokens) {
  const raw = tokens && tokens.length > 0 ? tokens : [0];
  /** @type {number[]} */
  const resolved = [];
  for (const t of raw) {
    const idx = t < 0 ? numTokens + t : t;
    if (idx >= 0 && idx < numTokens) {
      resolved.push(idx);
    }
  }
  return resolved;
}

/**
 * @param {import('../../config/schema/index.js').TraceCategory} category
 * @param {number | undefined} layerIdx
 * @returns {(message: string) => void}
 */
function getTraceLogger(category, layerIdx) {
  switch (category) {
    case 'attn':
      return (/** @type {string} */ message) => /** @type {(layerIdx: number, message: string) => void} */ (trace.attn)(layerIdx ?? -1, message);
    case 'ffn':
      return (/** @type {string} */ message) => /** @type {(layerIdx: number, message: string) => void} */ (trace.ffn)(layerIdx ?? -1, message);
    case 'kv':
      return (/** @type {string} */ message) => /** @type {(layerIdx: number, message: string) => void} */ (trace.kv)(layerIdx ?? -1, message);
    case 'loader':
      return (/** @type {string} */ message) => trace.loader(message);
    case 'kernels':
      return (/** @type {string} */ message) => trace.kernels(message);
    case 'logits':
      return (/** @type {string} */ message) => trace.logits(message);
    case 'embed':
      return (/** @type {string} */ message) => trace.embed(message);
    case 'sample':
      return (/** @type {string} */ message) => trace.sample(message);
    case 'buffers':
      return (/** @type {string} */ message) => trace.buffers(message);
    case 'perf':
      return (/** @type {string} */ message) => trace.perf(message);
    case 'all':
    default: {
      return (/** @type {string} */ message) => trace.embed(message);
    }
  }
}

/**
 * Run configured probes for a specific stage.
 * @param {import('../../config/schema/index.js').ProbeStage} stage
 * @param {GPUBuffer | Float32Array} buffer
 * @param {{ layerIdx?: number; numTokens: number; hiddenSize: number; probes?: import('../../config/schema/index.js').ProbeConfigSchema[] | null; recorder?: import('../../gpu/command-recorder.js').CommandRecorder | null }} options
 * @returns {Promise<void>}
 */
export async function runProbes(stage, buffer, options) {
  const { layerIdx, numTokens, hiddenSize, probes, recorder } = options;
  if (!probes || probes.length === 0) return;
  if (!buffer) return;
  if (recorder) return;

  const isCpuBuffer = buffer instanceof Float32Array;
  const device = isCpuBuffer ? null : getDevice();
  if (!isCpuBuffer && !device) return;

  const stageProbes = probes.filter((probe) => probe.stage === stage);
  if (stageProbes.length === 0) return;
  if (!isCpuBuffer && !allowReadback(`probe.${stage}`)) return;

  for (const probe of stageProbes) {
    if (!matchesLayer(probe.layers, layerIdx)) continue;

    const dims = probe.dims ?? [];
    if (dims.length === 0) continue;

    const tokens = resolveTokens(probe.tokens, numTokens);
    if (tokens.length === 0) continue;

    const category = probe.category && probe.category !== 'all'
      ? probe.category
      : STAGE_DEFAULT_CATEGORY[stage];
    const logger = getTraceLogger(category, layerIdx);
    const probeId = probe.id ? ` ${probe.id}` : '';

    for (const tokenIdx of tokens) {
      /** @type {string[]} */
      const values = [];
      for (const dimIdx of dims) {
        if (dimIdx < 0 || dimIdx >= hiddenSize) {
          values.push(`${dimIdx}=out_of_range`);
          continue;
        }
        if (isCpuBuffer) {
          const idx = tokenIdx * hiddenSize + dimIdx;
          const value = /** @type {Float32Array} */ (buffer)[idx];
          values.push(`${dimIdx}=${value.toFixed(4)}`);
          continue;
        }
        const offset = (tokenIdx * hiddenSize + dimIdx) * 4;
        const staging = /** @type {GPUDevice} */ (device).createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = /** @type {GPUDevice} */ (device).createCommandEncoder();
        enc.copyBufferToBuffer(/** @type {GPUBuffer} */ (buffer), offset, staging, 0, 4);
        /** @type {GPUDevice} */ (device).queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const value = new Float32Array(staging.getMappedRange().slice(0))[0];
        staging.unmap();
        staging.destroy();
        values.push(`${dimIdx}=${value.toFixed(4)}`);
      }

      logger(`PROBE${probeId} stage=${stage} token=${tokenIdx} values=[${values.join(', ')}]`);
    }
  }
}

/**
 * @param {import('../../config/schema/index.js').ProbeConfigSchema[] | null | undefined} probes
 * @param {import('../../config/schema/index.js').ProbeStage} stage
 * @param {number | undefined} [layerIdx]
 * @returns {boolean}
 */
export function hasProbeStage(probes, stage, layerIdx) {
  if (!probes || probes.length === 0) return false;
  return probes.some((probe) => probe.stage === stage && matchesLayer(probe.layers, layerIdx));
}
