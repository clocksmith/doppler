

import { trace } from '../../../debug/index.js';
import { resolveCapturePolicy } from '../../../debug/capture-policy.js';
import { snapshotTensor, snapshotFromArray } from '../../../debug/tensor.js';
import { getDevice } from '../../../gpu/device.js';
import { allowReadback } from '../../../gpu/perf-guards.js';
import { f16ToF32 } from '../../../loader/dtype-utils.js';
import { readBufferSlice } from '../../../memory/buffer-pool.js';
import { PROBE_TO_CANONICAL } from './stage-names.js';
import { buildOpId } from './operator-identity.js';
import { getOperatorClass } from './stage-names.js';
import { getDriftPolicyId } from './drift-policy.js';


const STAGE_DEFAULT_CATEGORY = {
  embed_out: 'embed',
  // Attention stages (per-layer)
  attn_input: 'attn',
  post_input_norm: 'attn',
  attn_normed: 'attn',
  linear_qkv_proj: 'attn',
  linear_z_proj: 'attn',
  linear_a_proj: 'attn',
  linear_b_proj: 'attn',
  linear_core_out: 'attn',
  q_proj: 'attn',
  k_proj: 'attn',
  v_proj: 'attn',
  q_norm: 'attn',
  k_norm: 'attn',
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


function matchesLayer(layers, layerIdx) {
  if (!layers || layers.length === 0) return true;
  if (layerIdx === undefined || layerIdx === null) return false;
  return layers.includes(layerIdx);
}


function resolveTokens(tokens, numTokens) {
  const raw = tokens && tokens.length > 0 ? tokens : [0];
  
  const resolved = [];
  for (const t of raw) {
    const idx = t < 0 ? numTokens + t : t;
    if (idx >= 0 && idx < numTokens) {
      resolved.push(idx);
    }
  }
  return resolved;
}


function getTraceLogger(category, layerIdx) {
  switch (category) {
    case 'attn':
      return ( message) =>  (trace.attn)(layerIdx ?? -1, message);
    case 'ffn':
      return ( message) =>  (trace.ffn)(layerIdx ?? -1, message);
    case 'kv':
      return ( message) =>  (trace.kv)(layerIdx ?? -1, message);
    case 'loader':
      return ( message) => trace.loader(message);
    case 'kernels':
      return ( message) => trace.kernels(message);
    case 'logits':
      return ( message) => trace.logits(message);
    case 'embed':
      return ( message) => trace.embed(message);
    case 'sample':
      return ( message) => trace.sample(message);
    case 'buffers':
      return ( message) => trace.buffers(message);
    case 'perf':
      return ( message) => trace.perf(message);
    case 'all':
    default: {
      return ( message) => trace.embed(message);
    }
  }
}


export async function runProbes(stage, buffer, options) {
  const { layerIdx, numTokens, hiddenSize, probes, recorder, dtype = 'f32' } = options;
  if (!buffer) return;
  if (recorder && !options.operatorDiagnostics?.enabled) return;

  const isCpuBuffer = buffer instanceof Float32Array;
  const device = isCpuBuffer ? null : getDevice();
  if (!isCpuBuffer && !device) return;

  const diagnostics = options.operatorDiagnostics ?? null;
  const canonicalStage = getCanonicalStageName(stage);
  if (diagnostics?.enabled && diagnostics.emitter && canonicalStage) {
    const opId = buildOpId(canonicalStage, layerIdx);
    const operatorClass = getOperatorClass(canonicalStage);
    const captureLevel = resolveCapturePolicy(opId, diagnostics.captureConfig);
    const capture = await buildDiagnosticCapture(captureLevel, buffer, {
      isCpuBuffer,
      shape: [numTokens, hiddenSize],
      dtype,
    });
    diagnostics.emitter.emitRecord(canonicalStage, {
      layerIdx,
      phase: options.phase ?? null,
      tokenIndex: options.tokenIndex ?? null,
      dtype,
      shapeSignature: `${numTokens}x${hiddenSize}`,
      opType: operatorClass,
      capturePolicy: captureLevel,
      driftPolicyId: getDriftPolicyId(operatorClass),
      capture,
      captureArtifactIds: capture ? [`${opId}:capture`] : [],
    });
  }

  if (!probes || probes.length === 0) return;
  if (recorder) return;

  const stageProbes = probes.filter((probe) => probe.stage === stage);
  if (stageProbes.length === 0) return;
  if (!isCpuBuffer && !allowReadback(`probe.${stage}`)) return;

  // Determine bytes per element based on dtype
  const bytesPerElement = dtype === 'f16' ? 2 : 4;

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

      const values = [];
      for (const dimIdx of dims) {
        if (dimIdx < 0 || dimIdx >= hiddenSize) {
          values.push(`${dimIdx}=out_of_range`);
          continue;
        }
        if (isCpuBuffer) {
          const idx = tokenIdx * hiddenSize + dimIdx;
          const value =  (buffer)[idx];
          values.push(`${dimIdx}=${value.toFixed(4)}`);
          continue;
        }
        const elementOffset = tokenIdx * hiddenSize + dimIdx;
        const byteOffset = elementOffset * bytesPerElement;
        // WebGPU requires offset and size to be multiples of 4
        const alignedOffset = Math.floor(byteOffset / 4) * 4;
        const offsetWithinRead = byteOffset - alignedOffset;
        const readSize = 4; // Always read 4 bytes (aligned)
        const readback = await readBufferSlice(buffer, alignedOffset, readSize);
        let value;
        if (dtype === 'f16') {
          // offsetWithinRead is 0 or 2 for F16 - extract correct u16
          const u16Array = new Uint16Array(readback);
          const u16Index = offsetWithinRead / 2;
          value = f16ToF32(u16Array[u16Index]);
        } else {
          value = new Float32Array(readback)[0];
        }
        values.push(`${dimIdx}=${value.toFixed(4)}`);
      }

      logger(`PROBE${probeId} stage=${stage} token=${tokenIdx} values=[${values.join(', ')}]`);
    }
  }
}


export function hasProbeStage(probes, stage, layerIdx) {
  if (!probes || probes.length === 0) return false;
  return probes.some((probe) => probe.stage === stage && matchesLayer(probe.layers, layerIdx));
}


export function getCanonicalStageName(probeStageName) {
  return PROBE_TO_CANONICAL[probeStageName] ?? null;
}

async function buildDiagnosticCapture(level, buffer, options) {
  if (level === 'none') return null;

  const { isCpuBuffer, shape, dtype } = options;
  const snapshot = isCpuBuffer
    ? snapshotFromArray(buffer, shape, dtype)
    : await snapshotTensor(buffer, shape, dtype);
  if (!snapshot?.ok) {
    return {
      level,
      error: snapshot?.error ?? 'capture_failed',
      shape,
      dtype,
      sample: null,
      stats: null,
    };
  }

  return {
    level,
    shape: snapshot.shape,
    dtype: snapshot.dtype,
    sample: Array.isArray(snapshot.sample) ? snapshot.sample : null,
    stats: snapshot.stats ?? null,
    hasNaN: snapshot.hasNaN === true,
    hasInf: snapshot.hasInf === true,
  };
}
