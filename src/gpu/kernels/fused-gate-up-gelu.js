// JS dispatcher for fused_gate_up_gelu_gated_f16.wgsl.
// Computes output[M, N] = gelu_tanh(input @ W_gate) * (input @ W_up) in one pass.
// N = intermediate_size (per-layer; handles Gemma 4's double-wide MLP via uniform).

import { getDevice } from '../device.js';
import { acquireBuffer } from '../../memory/buffer-pool.js';
import { createTensor } from '../tensor.js';
import { KernelBase } from './kernel-base.js';
import { createUniformBufferWithView } from './utils.js';
import { releaseUniformBuffer } from '../uniform-cache.js';
import { getBuffer } from '../weight-buffer.js';

// Tile geometry tuned 2026-04-17 for RDNA3 (Radeon 8060S) + Gemma 4 E2B-scale
// FFN shapes (hidden=1536, intermediate=6144). COLS_PER_WG=64 keeps the total
// workgroup launch count in the low thousands for prefill M=64 — the smaller
// COLS_PER_WG=8 default used 49k workgroups and was occupancy-starved by
// launch/retire overhead.
const WORKGROUP_SIZE = 256;
const COLS_PER_WG = 64;

class FusedGateUpGeluKernel extends KernelBase {
  async getPipeline(variant, constants = null) {
    return this.getPipelineFor('matmul', variant, null, constants);
  }
  dispatch(pipeline, bindGroup, workgroups) {
    this.dispatchKernel(pipeline, bindGroup, workgroups, 'fused_gate_up_gelu');
  }
  record(recorder, pipeline, bindGroup, workgroups) {
    this.recordKernel(recorder, pipeline, bindGroup, workgroups, 'fused_gate_up_gelu');
  }
}

let cachedKernel = null;
function getKernel(device) {
  if (!cachedKernel) cachedKernel = new FusedGateUpGeluKernel(device);
  return cachedKernel;
}

function createFusedGateUpGeluUniform(device, recorder, params) {
  return createUniformBufferWithView(
    'fused_gate_up_gelu_uniforms',
    32,
    (view) => {
      view.setUint32(0, params.M, true);
      view.setUint32(4, params.hiddenSize, true);
      view.setUint32(8, params.intermediateSize, true);
      view.setUint32(12, params.transposeB ? 1 : 0, true);
      view.setUint32(16, 0, true);
      view.setUint32(20, 0, true);
      view.setUint32(24, 0, true);
      view.setUint32(28, 0, true);
    },
    recorder,
    device
  );
}

async function executeFusedGateUpGelu(recorder, input, wGate, wUp, options) {
  const {
    M,
    hiddenSize,
    intermediateSize,
    transposeB = true,
    outputBuffer = null,
  } = options;

  if (!Number.isFinite(M) || !Number.isFinite(hiddenSize) || !Number.isFinite(intermediateSize)) {
    throw new Error('[FusedGateUpGelu] M, hiddenSize, intermediateSize must be finite.');
  }
  if (input.dtype !== 'f16') {
    throw new Error('[FusedGateUpGelu] input must be f16.');
  }

  const device = getDevice();
  if (!device) throw new Error('[FusedGateUpGelu] No GPU device.');

  const kernel = getKernel(device);
  const constants = { WORKGROUP_SIZE, COLS_PER_WG };
  const pipeline = await kernel.getPipeline('gate_up_gelu_gated_tiled_f16', constants);

  const outputSize = M * intermediateSize * 2;  // f16
  const output = outputBuffer || acquireBuffer(outputSize, undefined, 'fused_gate_up_gelu_output');

  const uniform = createFusedGateUpGeluUniform(device, recorder, {
    M, hiddenSize, intermediateSize, transposeB,
  });
  const wGateBuf = getBuffer(wGate);
  const wUpBuf = getBuffer(wUp);

  const bindGroup = device.createBindGroup({
    label: 'fused_gate_up_gelu_bg',
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniform } },
      { binding: 1, resource: { buffer: input.buffer } },
      { binding: 2, resource: { buffer: wGateBuf } },
      { binding: 3, resource: { buffer: wUpBuf } },
      { binding: 4, resource: { buffer: output } },
    ],
  });

  const colTiles = Math.ceil(intermediateSize / COLS_PER_WG);
  const workgroups = [colTiles, M, 1];

  if (recorder) {
    kernel.record(recorder, pipeline, bindGroup, workgroups);
  } else {
    kernel.dispatch(pipeline, bindGroup, workgroups);
  }

  if (uniform) releaseUniformBuffer(uniform);
  return createTensor(output, 'f16', [M, intermediateSize], 'fused_gate_up_gelu_output');
}

export async function runFusedGateUpGelu(input, wGate, wUp, options = {}) {
  return executeFusedGateUpGelu(null, input, wGate, wUp, options);
}

export async function recordFusedGateUpGelu(recorder, input, wGate, wUp, options = {}) {
  return executeFusedGateUpGelu(recorder, input, wGate, wUp, options);
}
