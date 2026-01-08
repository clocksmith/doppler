/**
 * Config-driven probe helpers for targeted GPU buffer inspection.
 *
 * Probes let us read specific token/dimension values without adding
 * ad-hoc logs in the code. Probes are configured via runtime.debug.probes.
 */

import type { ProbeConfigSchema, ProbeStage, TraceCategory } from '../../config/schema/index.js';
import { trace } from '../../debug/index.js';
import { getDevice } from '../../gpu/device.js';
import { allowReadback } from '../../gpu/perf-guards.js';
import type { CommandRecorder } from '../../gpu/command-recorder.js';

type TraceLogger = (message: string) => void;
type TraceLoggerWithLayer = (layerIdx: number, message: string) => void;

const STAGE_DEFAULT_CATEGORY: Record<ProbeStage, TraceCategory> = {
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

function matchesLayer(layers: number[] | null | undefined, layerIdx?: number): boolean {
  if (!layers || layers.length === 0) return true;
  if (layerIdx === undefined || layerIdx === null) return false;
  return layers.includes(layerIdx);
}

function resolveTokens(tokens: number[] | null | undefined, numTokens: number): number[] {
  const raw = tokens && tokens.length > 0 ? tokens : [0];
  const resolved: number[] = [];
  for (const t of raw) {
    const idx = t < 0 ? numTokens + t : t;
    if (idx >= 0 && idx < numTokens) {
      resolved.push(idx);
    }
  }
  return resolved;
}

function getTraceLogger(category: TraceCategory, layerIdx?: number): TraceLogger {
  switch (category) {
    case 'attn':
      return (message: string) => (trace.attn as TraceLoggerWithLayer)(layerIdx ?? -1, message);
    case 'ffn':
      return (message: string) => (trace.ffn as TraceLoggerWithLayer)(layerIdx ?? -1, message);
    case 'kv':
      return (message: string) => (trace.kv as TraceLoggerWithLayer)(layerIdx ?? -1, message);
    case 'loader':
      return (message: string) => trace.loader(message);
    case 'kernels':
      return (message: string) => trace.kernels(message);
    case 'logits':
      return (message: string) => trace.logits(message);
    case 'embed':
      return (message: string) => trace.embed(message);
    case 'sample':
      return (message: string) => trace.sample(message);
    case 'buffers':
      return (message: string) => trace.buffers(message);
    case 'perf':
      return (message: string) => trace.perf(message);
    case 'all':
    default: {
      return (message: string) => trace.embed(message);
    }
  }
}

/**
 * Run configured probes for a specific stage.
 */
export async function runProbes(
  stage: ProbeStage,
  buffer: GPUBuffer | Float32Array,
  options: {
    layerIdx?: number;
    numTokens: number;
    hiddenSize: number;
    probes?: ProbeConfigSchema[] | null;
    recorder?: CommandRecorder | null;
  }
): Promise<void> {
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
      const values: string[] = [];
      for (const dimIdx of dims) {
        if (dimIdx < 0 || dimIdx >= hiddenSize) {
          values.push(`${dimIdx}=out_of_range`);
          continue;
        }
        if (isCpuBuffer) {
          const idx = tokenIdx * hiddenSize + dimIdx;
          const value = (buffer as Float32Array)[idx];
          values.push(`${dimIdx}=${value.toFixed(4)}`);
          continue;
        }
        const offset = (tokenIdx * hiddenSize + dimIdx) * 4;
        const staging = device!.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = device!.createCommandEncoder();
        enc.copyBufferToBuffer(buffer as GPUBuffer, offset, staging, 0, 4);
        device!.queue.submit([enc.finish()]);
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

export function hasProbeStage(
  probes: ProbeConfigSchema[] | null | undefined,
  stage: ProbeStage,
  layerIdx?: number
): boolean {
  if (!probes || probes.length === 0) return false;
  return probes.some((probe) => probe.stage === stage && matchesLayer(probe.layers, layerIdx));
}
