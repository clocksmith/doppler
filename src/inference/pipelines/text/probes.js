

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
import { maybeWriteFixtureSnapshot } from './tsir-fixture-writer.js';


const STAGE_DEFAULT_CATEGORY = {
  embed_out: 'embed',
  per_layer_embed_out: 'embed',
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
  v_norm: 'attn',
  q_rope: 'attn',
  k_rope: 'attn',
  attn_scores: 'attn',
  attn_core_out: 'attn',
  attn_out: 'attn',
  o_proj: 'attn',
  post_attn: 'attn',
  // FFN stages (per-layer)
  per_layer_projection_in: 'ffn',
  per_layer_projection_scaled: 'ffn',
  per_layer_input: 'ffn',
  per_layer_input_gate: 'ffn',
  per_layer_input_activation: 'ffn',
  per_layer_input_projection: 'ffn',
  post_per_layer_input_norm: 'ffn',
  post_per_layer_input: 'ffn',
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
  // Skip when a recorder is in flight unless operatorDiagnostics is
  // actively capturing (diagnose mode emitter or tsirFixture writer).
  if (recorder
    && !options.operatorDiagnostics?.enabled
    && !options.operatorDiagnostics?.tsirFixture?.dir) return;

  const isCpuBuffer = buffer instanceof Float32Array;
  const device = isCpuBuffer ? null : getDevice();
  if (!isCpuBuffer && !device) return;

  const diagnostics = options.operatorDiagnostics ?? null;
  // TSIR fixture writer: when operatorDiagnostics.tsirFixture.dir is
  // set and the stage maps to one of the four TSIR boundary points
  // (post_input_norm/linear_qkv_proj/post_attn/layer_out), capture
  // the activation tensor as a .npy file under
  // <dir>/layer_<layerIdx>/<tsirStage>.npy. Builds the rung-5 frozen
  // Doppler reference fixture that the Doe rung ladder consumes.
  if (diagnostics?.tsirFixture?.dir) {
    const tsirRecord = await maybeWriteFixtureSnapshot(stage, buffer, {
      tsirFixture: diagnostics.tsirFixture,
      layerIdx,
      numTokens,
      hiddenSize,
      dtype,
      recorder,
    });
    if (tsirRecord) {
      const recordList = (diagnostics.tsirFixture.records ??= []);
      recordList.push(tsirRecord);
    }
  }
  const canonicalStage = getCanonicalStageName(stage);
  if (diagnostics?.enabled && diagnostics.emitter && canonicalStage) {
    const opId = buildOpId(canonicalStage, layerIdx);
    const operatorClass = getOperatorClass(canonicalStage);
    const captureLevel = resolveCapturePolicy(opId, diagnostics.captureConfig);
    const capture = await buildDiagnosticCapture(captureLevel, buffer, {
      isCpuBuffer,
      shape: [numTokens, hiddenSize],
      dtype,
      recorder,
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


function getCanonicalStageName(probeStageName) {
  return PROBE_TO_CANONICAL[probeStageName] ?? null;
}

async function buildDiagnosticCapture(level, buffer, options) {
  if (level === 'none') return null;

  const { isCpuBuffer, shape, dtype, recorder } = options;
  const includeData = level === 'full';
  if (!isCpuBuffer && recorder) {
    return createDeferredDiagnosticCapture(level, buffer, {
      recorder,
      shape,
      dtype,
    });
  }
  const snapshot = isCpuBuffer
    ? snapshotFromArray(buffer, shape, dtype, { includeData })
    : await snapshotTensor(buffer, shape, dtype, { includeData });
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
    data: Array.isArray(snapshot.data) ? snapshot.data : undefined,
  };
}

function createDeferredDiagnosticCapture(level, buffer, options) {
  const { recorder, shape, dtype } = options;
  const elementSize = dtype === 'f16' ? 2 : 4;
  const numElements = (shape ?? []).reduce((a, b) => a * b, 1);
  const readSize = numElements > 0
    ? Math.min(buffer.size, numElements * elementSize)
    : buffer.size;
  const staging = recorder.device.createBuffer({
    label: `${recorder.label}_diagnostic_capture`,
    size: readSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  recorder.getEncoder().copyBufferToBuffer(buffer, 0, staging, 0, readSize);

  const capture = {
    level,
    shape,
    dtype,
    sample: null,
    stats: null,
    hasNaN: false,
    hasInf: false,
  };

  recorder.enqueueCompletionTask(async () => {
    let mapped = false;
    try {
      await staging.mapAsync(GPUMapMode.READ);
      mapped = true;
      const snapshot = snapshotFromArray(
        decodeSnapshotBytes(staging.getMappedRange().slice(0), dtype),
        shape,
        dtype,
        { includeData: level === 'full' }
      );
      capture.shape = snapshot.shape;
      capture.dtype = snapshot.dtype;
      capture.sample = Array.isArray(snapshot.sample) ? snapshot.sample : null;
      capture.stats = snapshot.stats ?? null;
      capture.hasNaN = snapshot.hasNaN === true;
      capture.hasInf = snapshot.hasInf === true;
      capture.data = Array.isArray(snapshot.data) ? snapshot.data : undefined;
    } catch (error) {
      capture.error = error instanceof Error ? error.message : String(error);
      capture.sample = null;
      capture.stats = null;
    } finally {
      if (mapped) {
        staging.unmap();
      }
      staging.destroy();
    }
  });

  return capture;
}

function decodeSnapshotBytes(data, dtype) {
  if (dtype === 'f16') {
    const src = new Uint16Array(data);
    const decoded = new Float32Array(src.length);
    for (let i = 0; i < src.length; i++) {
      decoded[i] = f16ToF32(src[i]);
    }
    return decoded;
  }
  return new Float32Array(data);
}
