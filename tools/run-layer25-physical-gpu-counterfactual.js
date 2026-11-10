#!/usr/bin/env node
// Physical WebGPU execution of Flash-Next Layer 25 stage witnesses and
// substituted-input counterfactuals (R0_physical_gpu, R1_physical_gpu, R2_physical_gpu).
// Dispatches the actual production WGSL shaders on a physical GPU via Chromium WebGPU
// and verifies 0-ULP agreement against both retained GPU captures and the offline JS WGSL witness.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createServer } from 'node:http';
import { execFileSync } from 'node:child_process';
import { chromium } from 'playwright';
import { readNpyF32 } from './compare-source-boundaries.js';
import { halfValue, dotEnvelope } from './numerical-discontinuity-core.js';
import {
  witnessHyperConnectionNorm,
  witnessLowRankSilu,
  witnessMixSigmoid,
  witnessInjectGate,
  witnessHyperConnectionBranch,
  witnessHyperConnectionInject,
  witnessMatmulRowSubgroupWave64,
  witnessMatmulF16wF32aTiled16,
  witnessMatmulF16wF32aSeq,
  witnessGatedDeltaCore,
  compareF32Bits,
} from './layer25-wgsl-witness-core.js';

const [protocolPath, artifactDir, qualifyDir, layer25TraceDir, routerTraceDir, shaderDir, outputPath] =
  process.argv.slice(2);
if (!outputPath) {
  throw new Error(
    'Usage: run-layer25-physical-gpu-counterfactual.js protocol.json artifactDir qualifyDir layer25TraceDir routerTraceDir shaderDir output.json'
  );
}

const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const f32Sha = (arr) => sha(Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
const protocolBytes = fs.readFileSync(protocolPath);
const p = JSON.parse(protocolBytes);
if (p.schema !== 'doppler.layer25-prerouter-witness-protocol/v1' || p.qualificationAuthority !== false) {
  throw new Error('Unsupported or non-diagnostic protocol schema');
}

const identities = [];
function checked(file, expected, label = path.basename(file)) {
  const bytes = fs.readFileSync(file);
  const actual = sha(bytes);
  if (!expected || actual !== expected) {
    throw new Error(`Identity mismatch: ${label} (expected ${expected}, got ${actual})`);
  }
  identities.push({ file: label, sha256: actual, bytes: bytes.length });
  return bytes;
}

function readShardRange(file, offset, size) {
  const fd = fs.openSync(file, 'r');
  try {
    const bytes = Buffer.alloc(size);
    let count = 0;
    while (count < size) {
      const n = fs.readSync(fd, bytes, count, size - count, offset + count);
      if (!n) throw new Error(`Truncated weight span in ${path.basename(file)}`);
      count += n;
    }
    return bytes;
  } finally {
    fs.closeSync(fd);
  }
}

const manifest = JSON.parse(checked(path.join(artifactDir, 'manifest.json'), p.manifestSha256, 'manifest.json'));
const stagesObs = JSON.parse(
  checked(path.join(qualifyDir, p.stagesObservation), p.stagesObservationSha256, p.stagesObservation)
);
const routerObs = JSON.parse(
  checked(path.join(qualifyDir, p.routerObservation), p.routerObservationSha256, p.routerObservation)
);
const routerCmp = JSON.parse(
  checked(path.join(qualifyDir, p.routerComparison), p.routerComparisonSha256, p.routerComparison)
);
const l25Trace = JSON.parse(
  checked(path.join(layer25TraceDir, p.layer25TraceReceipt), p.layer25TraceReceiptSha256, `layer25/${p.layer25TraceReceipt}`)
);
const routerTrace = JSON.parse(
  checked(path.join(routerTraceDir, p.routerTraceReceipt), p.routerTraceReceiptSha256, `router/${p.routerTraceReceipt}`)
);

assert(stagesObs.diagnosticCompleted && routerObs.completed && l25Trace.diagnosticCompleted && routerTrace.completed);
assert.equal(routerCmp.firstChangedLayer, p.layer);

const weightPrefix = `model.language_model.layers.${p.layer}.`;
const weightIdentities = [];
function loadWeightRawAndF32(shortName) {
  const expected = p.weights[shortName];
  if (!expected) throw new Error(`Missing protocol weight entry: ${shortName}`);
  const desc = manifest.tensors[weightPrefix + shortName];
  if (!desc || desc.dtype !== expected.dtype || desc.shape.join(',') !== expected.shape.join(',')) {
    throw new Error(`Weight descriptor mismatch: ${shortName}`);
  }
  const spans = desc.spans ?? [{ shardIndex: desc.shard, offset: desc.offset, size: desc.size }];
  const raw = Buffer.concat(
    spans.map((s) => readShardRange(path.join(artifactDir, manifest.shards[s.shardIndex].filename), s.offset, s.size))
  );
  const actualSha = sha(raw);
  if (actualSha !== expected.sha256 || raw.length !== expected.bytes) {
    throw new Error(`Weight SHA-256 mismatch: ${shortName}`);
  }
  weightIdentities.push({ tensor: shortName, dtype: desc.dtype, shape: desc.shape, bytes: raw.length, sha256: actualSha });
  const count = raw.length / 2;
  const f32 = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    f32[i] = halfValue(raw.readUInt16LE(i * 2));
  }
  return { raw, f32 };
}

const rawW = {
  attnNorm: loadWeightRawAndF32('attn_hyper_connection.hc_norm.weight'),
  attnDown: loadWeightRawAndF32('attn_hyper_connection.input_mix_weight_down.weight'),
  attnUp: loadWeightRawAndF32('attn_hyper_connection.input_mix_weight_up.weight'),
  attnInject: loadWeightRawAndF32('attn_hyper_connection.block_inject_weight.weight'),
  mlpNorm: loadWeightRawAndF32('mlp_hyper_connection.hc_norm.weight'),
  mlpDown: loadWeightRawAndF32('mlp_hyper_connection.input_mix_weight_down.weight'),
  mlpUp: loadWeightRawAndF32('mlp_hyper_connection.input_mix_weight_up.weight'),
  mlpInject: loadWeightRawAndF32('mlp_hyper_connection.block_inject_weight.weight'),
  qkvProj: loadWeightRawAndF32('linear_attn.in_proj_qkv.weight'),
  zProj: loadWeightRawAndF32('linear_attn.in_proj_z.weight'),
  aProj: loadWeightRawAndF32('linear_attn.in_proj_a.weight'),
  bProj: loadWeightRawAndF32('linear_attn.in_proj_b.weight'),
  conv1d: loadWeightRawAndF32('linear_attn.conv1d.weight'),
  aLog: loadWeightRawAndF32('linear_attn.A_log'),
  dtBias: loadWeightRawAndF32('linear_attn.dt_bias'),
  normW: loadWeightRawAndF32('linear_attn.norm.weight'),
  outProj: loadWeightRawAndF32('linear_attn.out_proj.weight'),
  routerGate: loadWeightRawAndF32('mlp.gate.weight'),
};

const W = Object.fromEntries(Object.entries(rawW).map(([k, v]) => [k, v.f32]));

const shaderFiles = {
  hcNorm: 'hyper_connection_norm.wgsl',
  hcActivation: 'hyper_connection_activation.wgsl',
  hcBranch: 'hyper_connection_branch.wgsl',
  hcInject: 'hyper_connection_inject.wgsl',
  matmulRowSubgroup: 'matmul_f32_row_subgroup.wgsl',
  matmulF16Tiled: 'matmul_f16w_f32a_tiled.wgsl',
  matmulF16Seq: 'matmul_f16w_f32a.wgsl',
  gdnConv: 'gated_delta_conv.wgsl',
  gdnRecurrentSigmoid: 'gated_delta_recurrent_sigmoid_gate.wgsl',
};
const shaders = {};
const shaderIdentities = [];
for (const [key, filename] of Object.entries(shaderFiles)) {
  const fullPath = path.join(shaderDir, filename);
  const code = fs.readFileSync(fullPath, 'utf8');
  shaders[key] = code;
  shaderIdentities.push({ key, file: filename, sha256: sha(code), bytes: Buffer.byteLength(code) });
}

function bufferToF32(buf) {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

const gpuStages = {};
for (const entry of stagesObs.tensors) {
  const key = `${entry.role}.${entry.stage}`;
  const bytes = checked(path.join(qualifyDir, path.basename(entry.file)), entry.sha256, path.basename(entry.file));
  gpuStages[key] = bufferToF32(bytes);
}

const routerRecord = routerObs.rows[0].routerObservations.find((r) => r.layer === p.layer);
const gpuRouterInputAll = bufferToF32(
  checked(path.join(qualifyDir, path.basename(routerRecord.input.file)), routerRecord.input.sha256, path.basename(routerRecord.input.file))
);
const gpuRouterLogitsAll = bufferToF32(
  checked(path.join(qualifyDir, path.basename(routerRecord.logits.file)), routerRecord.logits.sha256, path.basename(routerRecord.logits.file))
);

const cpuStages = {};
for (const entry of l25Trace.tensors) {
  const key = `${entry.role}.${entry.stage}`;
  const bytes = checked(path.join(layer25TraceDir, entry.file), entry.sha256, `layer25/${entry.file}`);
  cpuStages[key] = readNpyF32(bytes).values;
}

const routerNpzName = `router-${p.layer}.npz`;
const routerNpzPath = path.join(routerTraceDir, routerNpzName);
checked(routerNpzPath, routerTrace.files[routerNpzName], `router/${routerNpzName}`);
const readRouterNpy = (name) =>
  readNpyF32(execFileSync('unzip', ['-p', routerNpzPath, `${name}.npy`], { maxBuffer: 40 * 1024 * 1024 })).values;
const cpuRouterInputAll = readRouterNpy('input');
const cpuRouterLogitsAll = readRouterNpy('logits');

const positions = p.positions;
const numPos = positions.length;
function slicePosRows(fullSeq, width) {
  const out = new Float32Array(numPos * width);
  for (let pi = 0; pi < numPos; pi++) {
    const pos = positions[pi];
    out.set(fullSeq.subarray(pos * width, (pos + 1) * width), pi * width);
  }
  return out;
}

gpuStages['router25.input'] = slicePosRows(gpuRouterInputAll, p.hiddenSize);
gpuStages['router25.logits'] = slicePosRows(gpuRouterLogitsAll, p.expertCount);
cpuStages['router25.input'] = slicePosRows(cpuRouterInputAll, p.hiddenSize);
cpuStages['router25.logits'] = slicePosRows(cpuRouterLogitsAll, p.expertCount);

// Compute offline JS WGSL reference results for R0, R1, R2 to compare with physical GPU
function runJsWgslHcForward(streamIn, normW, downW, upW) {
  const norm = witnessHyperConnectionNorm(streamIn, normW, numPos);
  const down = witnessMatmulRowSubgroupWave64(norm, numPos, p.streamWidth, downW, p.lowRankDim);
  const lowRank = witnessLowRankSilu(down, p.numStreams);
  const up = witnessMatmulRowSubgroupWave64(lowRank, numPos, p.lowRankDim, upW, p.streamWidth);
  const mix = witnessMixSigmoid(up);
  const branchInput = witnessHyperConnectionBranch(norm, mix, numPos, p.numStreams, p.hiddenSize);
  return { norm, down, lowRank, up, mix, branchInput };
}

function runJsWgslHcInject(streamIn, norm, branchOut, injectW) {
  const logits = witnessMatmulRowSubgroupWave64(norm, numPos, p.streamWidth, injectW, p.numStreams);
  const gate = witnessInjectGate(logits, p.numStreams);
  const injected = witnessHyperConnectionInject(streamIn, branchOut, gate, numPos, p.numStreams, p.hiddenSize);
  return { logits, gate, injected };
}

function runJsLinearAttnWithSplicedPosRows(baseQkv, baseZ, baseA, baseB, posBranchInput) {
  const seqQkv = new Float32Array(baseQkv);
  const seqZ = new Float32Array(baseZ);
  const seqA = new Float32Array(baseA);
  const seqB = new Float32Array(baseB);
  const posQkv = witnessMatmulF16wF32aTiled16(posBranchInput, numPos, p.hiddenSize, W.qkvProj, p.qkvDim);
  const posZ = witnessMatmulF16wF32aTiled16(posBranchInput, numPos, p.hiddenSize, W.zProj, p.zDim);
  const posA = witnessMatmulF16wF32aSeq(posBranchInput, numPos, p.hiddenSize, W.aProj, p.numValueHeads);
  const posB = witnessMatmulF16wF32aSeq(posBranchInput, numPos, p.hiddenSize, W.bProj, p.numValueHeads);
  for (let pi = 0; pi < numPos; pi++) {
    const pos = positions[pi];
    seqQkv.set(posQkv.subarray(pi * p.qkvDim, (pi + 1) * p.qkvDim), pos * p.qkvDim);
    seqZ.set(posZ.subarray(pi * p.zDim, (pi + 1) * p.zDim), pos * p.zDim);
    seqA.set(posA.subarray(pi * p.numValueHeads, (pi + 1) * p.numValueHeads), pos * p.numValueHeads);
    seqB.set(posB.subarray(pi * p.numValueHeads, (pi + 1) * p.numValueHeads), pos * p.numValueHeads);
  }
  const seqCore = witnessGatedDeltaCore(seqQkv, seqZ, seqA, seqB, W.conv1d, W.dtBias, W.aLog, W.normW, p.seqLen);
  const posCore = slicePosRows(seqCore, p.zDim);
  const branchOutput = witnessMatmulF16wF32aTiled16(posCore, numPos, p.zDim, W.outProj, p.hiddenSize);
  return { posQkv, posZ, posA, posB, seqCore, posCore, branchOutput };
}

const jsR0AttnHc = runJsWgslHcForward(gpuStages['attn_hc.input'], W.attnNorm, W.attnDown, W.attnUp);
const jsR0Lin = runJsLinearAttnWithSplicedPosRows(
  gpuStages['linear.attn.qkv_proj'],
  gpuStages['linear.attn.linear_z_proj'],
  gpuStages['linear.attn.linear_a_proj'],
  gpuStages['linear.attn.linear_b_proj'],
  jsR0AttnHc.branchInput
);
const jsR0Inject = runJsWgslHcInject(gpuStages['attn_hc.input'], jsR0AttnHc.norm, jsR0Lin.branchOutput, W.attnInject);
const jsR0MlpHc = runJsWgslHcForward(jsR0Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);
const jsR0RouterLogits = witnessMatmulRowSubgroupWave64(jsR0MlpHc.branchInput, numPos, p.hiddenSize, W.routerGate, p.expertCount);

const jsR1AttnHc = runJsWgslHcForward(cpuStages['attn_hc.input'], W.attnNorm, W.attnDown, W.attnUp);
const jsR1Inject = runJsWgslHcInject(cpuStages['attn_hc.input'], jsR1AttnHc.norm, cpuStages['attn_hc.branchOutput'], W.attnInject);
const jsR1MlpHc = runJsWgslHcForward(jsR1Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);
const jsR1RouterLogits = witnessMatmulRowSubgroupWave64(jsR1MlpHc.branchInput, numPos, p.hiddenSize, W.routerGate, p.expertCount);

const jsR2Lin = runJsLinearAttnWithSplicedPosRows(
  gpuStages['linear.attn.qkv_proj'],
  gpuStages['linear.attn.linear_z_proj'],
  gpuStages['linear.attn.linear_a_proj'],
  gpuStages['linear.attn.linear_b_proj'],
  jsR1AttnHc.branchInput
);
const jsR2Inject = runJsWgslHcInject(cpuStages['attn_hc.input'], jsR1AttnHc.norm, jsR2Lin.branchOutput, W.attnInject);
const jsR2MlpHc = runJsWgslHcForward(jsR2Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);
const jsR2RouterLogits = witnessMatmulRowSubgroupWave64(jsR2MlpHc.branchInput, numPos, p.hiddenSize, W.routerGate, p.expertCount);

// Serve binary blobs over localhost HTTP so Chromium WebGPU can fetch them without huge JSON strings
const blobs = new Map();
function registerBlob(name, typedOrBuf) {
  const buf = Buffer.isBuffer(typedOrBuf)
    ? typedOrBuf
    : Buffer.from(typedOrBuf.buffer, typedOrBuf.byteOffset, typedOrBuf.byteLength);
  blobs.set(`/${name}`, buf);
}

registerBlob('w_attnNorm.f32', W.attnNorm);
registerBlob('w_attnDown.f32', W.attnDown);
registerBlob('w_attnUp.f32', W.attnUp);
registerBlob('w_attnInject.f32', W.attnInject);
registerBlob('w_mlpNorm.f32', W.mlpNorm);
registerBlob('w_mlpDown.f32', W.mlpDown);
registerBlob('w_mlpUp.f32', W.mlpUp);
registerBlob('w_mlpInject.f32', W.mlpInject);
registerBlob('w_qkvProj.f16', rawW.qkvProj.raw);
registerBlob('w_zProj.f16', rawW.zProj.raw);
registerBlob('w_aProj.f16', rawW.aProj.raw);
registerBlob('w_bProj.f16', rawW.bProj.raw);
registerBlob('w_outProj.f16', rawW.outProj.raw);
registerBlob('w_conv1d.f32', W.conv1d);
registerBlob('w_dtBias.f32', W.dtBias);
registerBlob('w_aLog.f32', W.aLog);
registerBlob('w_normW.f32', W.normW);
registerBlob('w_routerGate.f32', W.routerGate);

registerBlob('gpu_attn_hc_input.f32', gpuStages['attn_hc.input']);
registerBlob('gpu_qkv_proj.f32', gpuStages['linear.attn.qkv_proj']);
registerBlob('gpu_z_proj.f32', gpuStages['linear.attn.linear_z_proj']);
registerBlob('gpu_a_proj.f32', gpuStages['linear.attn.linear_a_proj']);
registerBlob('gpu_b_proj.f32', gpuStages['linear.attn.linear_b_proj']);
registerBlob('cpu_attn_hc_input.f32', cpuStages['attn_hc.input']);
registerBlob('cpu_attn_hc_branchOutput.f32', cpuStages['attn_hc.branchOutput']);

const postedOutputs = new Map();
const server = createServer((req, res) => {
  if (req.method === 'POST' && req.url.startsWith('/out/')) {
    const chunks = [];
    req.on('data', (c) => chunks.push(c));
    req.on('end', () => {
      postedOutputs.set(req.url.slice(5), Buffer.concat(chunks));
      res.writeHead(200);
      res.end('ok');
    });
    return;
  }
  if (req.url === '/') {
    res.writeHead(200, { 'content-type': 'text/html' });
    res.end('<!doctype html><title>layer25-physical-gpu-counterfactual</title>');
    return;
  }
  const blob = blobs.get(req.url);
  if (blob) {
    res.writeHead(200, { 'content-type': 'application/octet-stream' });
    res.end(blob);
    return;
  }
  res.writeHead(404);
  res.end('not found');
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const baseUrl = `http://127.0.0.1:${server.address().port}`;

let browser;
let gpuMeta;
try {
  browser = await chromium.launch({
    channel: 'chrome',
    headless: true,
    args: [
      '--enable-unsafe-webgpu',
      ...(process.platform === 'darwin'
        ? ['--use-angle=metal']
        : ['--enable-features=Vulkan', '--use-angle=vulkan', '--disable-gpu-sandbox']),
    ],
  });
  const page = await browser.newPage();
  await page.goto(baseUrl);

  gpuMeta = await page.evaluate(
    async ({ shaders, params }) => {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter || adapter.info.isFallbackAdapter) {
        throw new Error('Physical WebGPU adapter required');
      }
      const requiredFeatures = ['subgroups', 'shader-f16'];
      for (const feat of requiredFeatures) {
        if (!adapter.features.has(feat)) {
          throw new Error(`Adapter missing required feature: ${feat}`);
        }
      }
      const device = await adapter.requestDevice({
        requiredFeatures,
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
          maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
          maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        },
      });
      const errors = [];
      device.addEventListener('uncapturederror', (e) => errors.push(e.error.message));
      const owned = [];
      let newGpuInvocations = 0;

      const makeBuffer = (size, usage, data) => {
        const aligned = Math.max(16, Math.ceil(size / 4) * 4);
        const buf = device.createBuffer({ size: aligned, usage });
        owned.push(buf);
        if (data) {
          const u8 = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
          device.queue.writeBuffer(buf, 0, u8);
        }
        return buf;
      };

      const fetchBytes = async (name) => {
        const res = await fetch(`/${name}`);
        if (!res.ok) throw new Error(`Failed to fetch /${name}`);
        return await res.arrayBuffer();
      };

      const postBytes = async (name, ab) => {
        const res = await fetch(`/out/${name}`, { method: 'POST', body: ab });
        if (!res.ok) throw new Error(`Failed to post /out/${name}`);
      };

      const readGpuBuffer = async (buf, byteLength) => {
        const staging = makeBuffer(byteLength, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
        const enc = device.createCommandEncoder();
        enc.copyBufferToBuffer(buf, 0, staging, 0, byteLength);
        device.queue.submit([enc.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const copy = staging.getMappedRange(0, byteLength).slice(0);
        staging.unmap();
        return copy;
      };

      const compile = async (code, entryPoint, constants = {}) => {
        const module = device.createShaderModule({ code });
        const info = await module.getCompilationInfo();
        const errs = info.messages.filter((m) => m.type === 'error');
        if (errs.length) throw new Error(errs.map((m) => m.message).join('\n'));
        return await device.createComputePipelineAsync({
          layout: 'auto',
          compute: { module, entryPoint, constants },
        });
      };

      try {
        const pipelines = {
          hcNorm: await compile(shaders.hcNorm, 'main', {
            WORKGROUP_SIZE: 256,
            STREAM_COUNT: params.numStreams,
            STREAM_WIDTH: params.hiddenSize,
          }),
          lowRankSilu: await compile(shaders.hcActivation, 'low_rank_silu', {
            WORKGROUP_SIZE: 256,
            STREAM_COUNT: params.numStreams,
            STREAM_WIDTH: 1,
          }),
          mixSigmoid: await compile(shaders.hcActivation, 'mix_sigmoid', {
            WORKGROUP_SIZE: 256,
            STREAM_COUNT: params.numStreams,
            STREAM_WIDTH: 1,
          }),
          injectGate: await compile(shaders.hcActivation, 'inject_gate', {
            WORKGROUP_SIZE: 256,
            STREAM_COUNT: params.numStreams,
            STREAM_WIDTH: 1,
          }),
          hcBranch: await compile(shaders.hcBranch, 'main', {
            WORKGROUP_SIZE: 256,
            STREAM_COUNT: params.numStreams,
            STREAM_WIDTH: params.hiddenSize,
          }),
          hcInject: await compile(shaders.hcInject, 'main', {
            WORKGROUP_SIZE: 256,
            STREAM_COUNT: params.numStreams,
            STREAM_WIDTH: params.hiddenSize,
          }),
          matmulRowSubgroup: await compile(shaders.matmulRowSubgroup, 'main', {
            WORKGROUP_SIZE: 128,
            COLS_PER_WG: 4,
          }),
          matmulF16Tiled: await compile(shaders.matmulF16Tiled, 'main', {}),
          matmulF16Seq: await compile(shaders.matmulF16Seq, 'main', { TILE_SIZE: 16 }),
          gdnConv: await compile(shaders.gdnConv, 'main', { WORKGROUP_SIZE: 256 }),
          gdnRecurrent: await compile(shaders.gdnRecurrentSigmoid, 'main', {
            WORKGROUP_SIZE: 128,
            RECURRENT_STATE_COMPONENTS: 1,
          }),
        };

        const dispatch = (pipeline, buffers, gx, gy = 1, gz = 1) => {
          const bg = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: buffers.map((b, i) => ({ binding: i, resource: { buffer: b } })),
          });
          const enc = device.createCommandEncoder();
          const pass = enc.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bg);
          pass.dispatchWorkgroups(gx, gy, gz);
          pass.end();
          device.queue.submit([enc.finish()]);
          newGpuInvocations += 1;
        };

        const runHcNormGpu = (inputBuf, weightBuf, rows) => {
          const outBytes = rows * params.streamWidth * 4;
          const outBuf = makeBuffer(outBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const u = new ArrayBuffer(16);
          const v = new DataView(u);
          const groups = rows * params.numStreams;
          v.setUint32(0, groups, true);
          v.setUint32(4, groups, true);
          v.setFloat32(8, params.rmsNormEps, true);
          v.setUint32(12, 0, true);
          const uBuf = makeBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          dispatch(pipelines.hcNorm, [uBuf, inputBuf, weightBuf, outBuf], groups, 1, 1);
          return outBuf;
        };

        const runMatmulRowSubgroupGpu = (aBuf, bBuf, M, N, K) => {
          const outBytes = M * N * 4;
          const outBuf = makeBuffer(outBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const wgX = Math.ceil(N / 4);
          const u = new ArrayBuffer(32);
          const v = new DataView(u);
          v.setUint32(0, M, true);
          v.setUint32(4, N, true);
          v.setUint32(8, K, true);
          v.setFloat32(12, 1.0, true);
          v.setUint32(16, 1, true);
          v.setUint32(20, wgX, true);
          const uBuf = makeBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          dispatch(pipelines.matmulRowSubgroup, [uBuf, aBuf, bBuf, outBuf], wgX, M, 1);
          return outBuf;
        };

        const runHcActivationGpu = (pipeline, inputBuf, count) => {
          const outBuf = makeBuffer(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const wgX = Math.ceil(count / 256);
          const u = new ArrayBuffer(16);
          const v = new DataView(u);
          v.setUint32(0, count, true);
          v.setUint32(4, wgX * 256, true);
          const uBuf = makeBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          dispatch(pipeline, [uBuf, inputBuf, outBuf], wgX, 1, 1);
          return outBuf;
        };

        const runHcBranchGpu = (normBuf, mixBuf, rows) => {
          const count = rows * params.hiddenSize;
          const outBuf = makeBuffer(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const wgX = Math.ceil(count / 256);
          const u = new ArrayBuffer(16);
          const v = new DataView(u);
          v.setUint32(0, count, true);
          v.setUint32(4, wgX * 256, true);
          const uBuf = makeBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          // Binding 1 is mix, Binding 2 is normed
          dispatch(pipelines.hcBranch, [uBuf, mixBuf, normBuf, outBuf], wgX, 1, 1);
          return outBuf;
        };

        const runHcInjectGpu = (streamsBuf, branchBuf, gateBuf, rows) => {
          const count = rows * params.streamWidth;
          const outBuf = makeBuffer(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const wgX = Math.ceil(count / 256);
          const u = new ArrayBuffer(16);
          const v = new DataView(u);
          v.setUint32(0, count, true);
          v.setUint32(4, wgX * 256, true);
          const uBuf = makeBuffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          dispatch(pipelines.hcInject, [uBuf, streamsBuf, branchBuf, gateBuf, outBuf], wgX, 1, 1);
          return outBuf;
        };

        const runMatmulF16TiledGpu = (aBuf, bF16Buf, M, N, K) => {
          const outBuf = makeBuffer(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const u = new ArrayBuffer(32);
          const v = new DataView(u);
          v.setUint32(0, M, true);
          v.setUint32(4, N, true);
          v.setUint32(8, K, true);
          v.setFloat32(12, 1.0, true);
          v.setUint32(16, 1, true);
          const uBuf = makeBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          dispatch(pipelines.matmulF16Tiled, [uBuf, aBuf, bF16Buf, outBuf], Math.ceil(M / 64), Math.ceil(N / 64), 1);
          return outBuf;
        };

        const runMatmulF16SeqGpu = (aBuf, bF16Buf, M, N, K) => {
          const outBuf = makeBuffer(M * N * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          const u = new ArrayBuffer(32);
          const v = new DataView(u);
          v.setUint32(0, M, true);
          v.setUint32(4, N, true);
          v.setUint32(8, K, true);
          v.setFloat32(12, 1.0, true);
          v.setUint32(16, 1, true);
          const uBuf = makeBuffer(32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);
          dispatch(pipelines.matmulF16Seq, [uBuf, aBuf, bF16Buf, outBuf], Math.ceil(M / 16), Math.ceil(N / 16), 1);
          return outBuf;
        };

        const runGdnCoreGpu = (qkvBuf, zBuf, aBuf, bBuf, convWBuf, dtBiasBuf, aLogBuf, normWBuf, numTokens) => {
          const u = new ArrayBuffer(64);
          const v = new DataView(u);
          v.setUint32(0, numTokens, true);
          v.setUint32(4, params.qkvDim, true);
          v.setUint32(8, params.convKernelSize, true);
          v.setUint32(12, params.numValueHeads, true);
          v.setUint32(16, params.numKeyHeads, true);
          v.setUint32(20, params.headKeyDim, true);
          v.setUint32(24, params.headValueDim, true);
          v.setUint32(28, params.numKeyHeads * params.headKeyDim, true);
          v.setUint32(32, params.numKeyHeads * params.headKeyDim, true);
          v.setUint32(36, params.zDim, true);
          v.setUint32(40, params.numValueHeads / params.numKeyHeads, true);
          v.setUint32(44, 0, true); // shared norm_mode
          v.setFloat32(48, params.rmsNormEps, true);
          v.setFloat32(52, 1e-6, true);
          v.setUint32(56, 0, true);
          v.setUint32(60, 0, true);
          const uBuf = makeBuffer(64, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, u);

          const convStateBytes = params.qkvDim * params.convKernelSize * 4;
          const convStateBuf = makeBuffer(
            convStateBytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            new Float32Array(params.qkvDim * params.convKernelSize)
          );
          const convOutBuf = makeBuffer(numTokens * params.qkvDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          dispatch(pipelines.gdnConv, [uBuf, qkvBuf, convWBuf, convStateBuf, convOutBuf], Math.ceil(params.qkvDim / 256), 1, 1);

          const recStateCount = params.numValueHeads * params.headKeyDim * params.headValueDim;
          const recStateBuf = makeBuffer(
            recStateCount * 4,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            new Float32Array(recStateCount)
          );
          const coreOutBuf = makeBuffer(numTokens * params.zDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          dispatch(
            pipelines.gdnRecurrent,
            [uBuf, convOutBuf, zBuf, aBuf, bBuf, dtBiasBuf, aLogBuf, normWBuf, recStateBuf, coreOutBuf],
            params.numValueHeads,
            1,
            1
          );
          return coreOutBuf;
        };

        // Upload shared weights once
        const wBufs = {
          attnNorm: makeBuffer(params.streamWidth * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_attnNorm.f32')),
          attnDown: makeBuffer(params.lowRankDim * params.streamWidth * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_attnDown.f32')),
          attnUp: makeBuffer(params.streamWidth * params.lowRankDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_attnUp.f32')),
          attnInject: makeBuffer(params.numStreams * params.streamWidth * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_attnInject.f32')),
          mlpNorm: makeBuffer(params.streamWidth * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_mlpNorm.f32')),
          mlpDown: makeBuffer(params.lowRankDim * params.streamWidth * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_mlpDown.f32')),
          mlpUp: makeBuffer(params.streamWidth * params.lowRankDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_mlpUp.f32')),
          mlpInject: makeBuffer(params.numStreams * params.streamWidth * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_mlpInject.f32')),
          qkvProj: makeBuffer(params.qkvDim * params.hiddenSize * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_qkvProj.f16')),
          zProj: makeBuffer(params.zDim * params.hiddenSize * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_zProj.f16')),
          aProj: makeBuffer(params.numValueHeads * params.hiddenSize * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_aProj.f16')),
          bProj: makeBuffer(params.numValueHeads * params.hiddenSize * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_bProj.f16')),
          outProj: makeBuffer(params.hiddenSize * params.zDim * 2, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_outProj.f16')),
          conv1d: makeBuffer(params.qkvDim * params.convKernelSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_conv1d.f32')),
          dtBias: makeBuffer(params.numValueHeads * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_dtBias.f32')),
          aLog: makeBuffer(params.numValueHeads * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_aLog.f32')),
          normW: makeBuffer(params.headValueDim * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_normW.f32')),
          routerGate: makeBuffer(params.expertCount * params.hiddenSize * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, await fetchBytes('w_routerGate.f32')),
        };

        const runHcForwardGpu = (streamInBuf, normW, downW, upW, rows) => {
          const norm = runHcNormGpu(streamInBuf, normW, rows);
          const down = runMatmulRowSubgroupGpu(norm, downW, rows, params.lowRankDim, params.streamWidth);
          const lowRank = runHcActivationGpu(pipelines.lowRankSilu, down, rows * params.lowRankDim);
          const up = runMatmulRowSubgroupGpu(lowRank, upW, rows, params.streamWidth, params.lowRankDim);
          const mix = runHcActivationGpu(pipelines.mixSigmoid, up, rows * params.streamWidth);
          const branchInput = runHcBranchGpu(norm, mix, rows);
          return { norm, down, lowRank, up, mix, branchInput };
        };

        const runHcInjectBlockGpu = (streamInBuf, normBuf, branchOutBuf, injectW, rows) => {
          const logits = runMatmulRowSubgroupGpu(normBuf, injectW, rows, params.numStreams, params.streamWidth);
          const gate = runHcActivationGpu(pipelines.injectGate, logits, rows * params.numStreams);
          const injected = runHcInjectGpu(streamInBuf, branchOutBuf, gate, rows);
          return { logits, gate, injected };
        };

        const baseQkvAb = await fetchBytes('gpu_qkv_proj.f32');
        const baseZAb = await fetchBytes('gpu_z_proj.f32');
        const baseAAb = await fetchBytes('gpu_a_proj.f32');
        const baseBAb = await fetchBytes('gpu_b_proj.f32');

        const runSplicedLinearAttnGpu = async (posBranchInputBuf) => {
          const posQkvBuf = runMatmulF16TiledGpu(posBranchInputBuf, wBufs.qkvProj, numPos, params.qkvDim, params.hiddenSize);
          const posZBuf = runMatmulF16TiledGpu(posBranchInputBuf, wBufs.zProj, numPos, params.zDim, params.hiddenSize);
          const posABuf = runMatmulF16SeqGpu(posBranchInputBuf, wBufs.aProj, numPos, params.numValueHeads, params.hiddenSize);
          const posBBuf = runMatmulF16SeqGpu(posBranchInputBuf, wBufs.bProj, numPos, params.numValueHeads, params.hiddenSize);

          const posQkv = new Float32Array(await readGpuBuffer(posQkvBuf, numPos * params.qkvDim * 4));
          const posZ = new Float32Array(await readGpuBuffer(posZBuf, numPos * params.zDim * 4));
          const posA = new Float32Array(await readGpuBuffer(posABuf, numPos * params.numValueHeads * 4));
          const posB = new Float32Array(await readGpuBuffer(posBBuf, numPos * params.numValueHeads * 4));

          const seqQkv = new Float32Array(baseQkvAb.slice(0));
          const seqZ = new Float32Array(baseZAb.slice(0));
          const seqA = new Float32Array(baseAAb.slice(0));
          const seqB = new Float32Array(baseBAb.slice(0));

          for (let pi = 0; pi < numPos; pi++) {
            const pos = params.positions[pi];
            seqQkv.set(posQkv.subarray(pi * params.qkvDim, (pi + 1) * params.qkvDim), pos * params.qkvDim);
            seqZ.set(posZ.subarray(pi * params.zDim, (pi + 1) * params.zDim), pos * params.zDim);
            seqA.set(posA.subarray(pi * params.numValueHeads, (pi + 1) * params.numValueHeads), pos * params.numValueHeads);
            seqB.set(posB.subarray(pi * params.numValueHeads, (pi + 1) * params.numValueHeads), pos * params.numValueHeads);
          }

          const seqQkvBuf = makeBuffer(seqQkv.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, seqQkv);
          const seqZBuf = makeBuffer(seqZ.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, seqZ);
          const seqABuf = makeBuffer(seqA.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, seqA);
          const seqBBuf = makeBuffer(seqB.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, seqB);

          const seqCoreBuf = runGdnCoreGpu(
            seqQkvBuf,
            seqZBuf,
            seqABuf,
            seqBBuf,
            wBufs.conv1d,
            wBufs.dtBias,
            wBufs.aLog,
            wBufs.normW,
            params.seqLen
          );
          const seqCoreAb = await readGpuBuffer(seqCoreBuf, params.seqLen * params.zDim * 4);
          const seqCore = new Float32Array(seqCoreAb);
          const posCore = new Float32Array(numPos * params.zDim);
          for (let pi = 0; pi < numPos; pi++) {
            const pos = params.positions[pi];
            posCore.set(seqCore.subarray(pos * params.zDim, (pos + 1) * params.zDim), pi * params.zDim);
          }
          const posCoreBuf = makeBuffer(posCore.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, posCore);
          const branchOutBuf = runMatmulF16TiledGpu(posCoreBuf, wBufs.outProj, numPos, params.hiddenSize, params.zDim);
          return { posQkvBuf, posZBuf, posABuf, posBBuf, seqCoreAb, posCoreBuf, branchOutBuf };
        };

        const numPos = params.positions.length;

        // 1. R0_physical_gpu: Actual GPU Layer-24 state through physical GPU Layer-25 operators
        const gpuAttnInBuf = makeBuffer(
          numPos * params.streamWidth * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          await fetchBytes('gpu_attn_hc_input.f32')
        );
        const r0AttnHc = runHcForwardGpu(gpuAttnInBuf, wBufs.attnNorm, wBufs.attnDown, wBufs.attnUp, numPos);
        const r0Lin = await runSplicedLinearAttnGpu(r0AttnHc.branchInput);
        const r0Inject = runHcInjectBlockGpu(gpuAttnInBuf, r0AttnHc.norm, r0Lin.branchOutBuf, wBufs.attnInject, numPos);
        const r0MlpHc = runHcForwardGpu(r0Inject.injected, wBufs.mlpNorm, wBufs.mlpDown, wBufs.mlpUp, numPos);
        const r0RouterLogits = runMatmulRowSubgroupGpu(r0MlpHc.branchInput, wBufs.routerGate, numPos, params.expertCount, params.hiddenSize);

        await postBytes('r0_attn_norm.f32', await readGpuBuffer(r0AttnHc.norm, numPos * params.streamWidth * 4));
        await postBytes('r0_attn_down.f32', await readGpuBuffer(r0AttnHc.down, numPos * params.lowRankDim * 4));
        await postBytes('r0_attn_lowRank.f32', await readGpuBuffer(r0AttnHc.lowRank, numPos * params.lowRankDim * 4));
        await postBytes('r0_attn_up.f32', await readGpuBuffer(r0AttnHc.up, numPos * params.streamWidth * 4));
        await postBytes('r0_attn_mix.f32', await readGpuBuffer(r0AttnHc.mix, numPos * params.streamWidth * 4));
        await postBytes('r0_attn_branchInput.f32', await readGpuBuffer(r0AttnHc.branchInput, numPos * params.hiddenSize * 4));
        await postBytes('r0_pos_qkv.f32', await readGpuBuffer(r0Lin.posQkvBuf, numPos * params.qkvDim * 4));
        await postBytes('r0_pos_z.f32', await readGpuBuffer(r0Lin.posZBuf, numPos * params.zDim * 4));
        await postBytes('r0_pos_a.f32', await readGpuBuffer(r0Lin.posABuf, numPos * params.numValueHeads * 4));
        await postBytes('r0_pos_b.f32', await readGpuBuffer(r0Lin.posBBuf, numPos * params.numValueHeads * 4));
        await postBytes('r0_seq_core.f32', r0Lin.seqCoreAb);
        await postBytes('r0_attn_branchOutput.f32', await readGpuBuffer(r0Lin.branchOutBuf, numPos * params.hiddenSize * 4));
        await postBytes('r0_attn_logits.f32', await readGpuBuffer(r0Inject.logits, numPos * params.numStreams * 4));
        await postBytes('r0_attn_gate.f32', await readGpuBuffer(r0Inject.gate, numPos * params.numStreams * 4));
        await postBytes('r0_attn_injected.f32', await readGpuBuffer(r0Inject.injected, numPos * params.streamWidth * 4));
        await postBytes('r0_mlp_norm.f32', await readGpuBuffer(r0MlpHc.norm, numPos * params.streamWidth * 4));
        await postBytes('r0_mlp_down.f32', await readGpuBuffer(r0MlpHc.down, numPos * params.lowRankDim * 4));
        await postBytes('r0_mlp_lowRank.f32', await readGpuBuffer(r0MlpHc.lowRank, numPos * params.lowRankDim * 4));
        await postBytes('r0_mlp_up.f32', await readGpuBuffer(r0MlpHc.up, numPos * params.streamWidth * 4));
        await postBytes('r0_mlp_mix.f32', await readGpuBuffer(r0MlpHc.mix, numPos * params.streamWidth * 4));
        await postBytes('r0_mlp_branchInput.f32', await readGpuBuffer(r0MlpHc.branchInput, numPos * params.hiddenSize * 4));
        await postBytes('r0_router_logits.f32', await readGpuBuffer(r0RouterLogits, numPos * params.expertCount * 4));

        // 2. R1_physical_gpu: Substituted CPU attn_hc.input + CPU linear_attn branchOutput through physical GPU HC + router
        const cpuAttnInBuf = makeBuffer(
          numPos * params.streamWidth * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          await fetchBytes('cpu_attn_hc_input.f32')
        );
        const cpuBranchOutBuf = makeBuffer(
          numPos * params.hiddenSize * 4,
          GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          await fetchBytes('cpu_attn_hc_branchOutput.f32')
        );
        const r1AttnHc = runHcForwardGpu(cpuAttnInBuf, wBufs.attnNorm, wBufs.attnDown, wBufs.attnUp, numPos);
        const r1Inject = runHcInjectBlockGpu(cpuAttnInBuf, r1AttnHc.norm, cpuBranchOutBuf, wBufs.attnInject, numPos);
        const r1MlpHc = runHcForwardGpu(r1Inject.injected, wBufs.mlpNorm, wBufs.mlpDown, wBufs.mlpUp, numPos);
        const r1RouterLogits = runMatmulRowSubgroupGpu(r1MlpHc.branchInput, wBufs.routerGate, numPos, params.expertCount, params.hiddenSize);

        await postBytes('r1_mlp_branchInput.f32', await readGpuBuffer(r1MlpHc.branchInput, numPos * params.hiddenSize * 4));
        await postBytes('r1_router_logits.f32', await readGpuBuffer(r1RouterLogits, numPos * params.expertCount * 4));

        // 3. R2_physical_gpu: Substituted CPU attn_hc.input at [1717, 2271, 2303] through 100% physical GPU Layer-25 operators
        const r2Lin = await runSplicedLinearAttnGpu(r1AttnHc.branchInput);
        const r2Inject = runHcInjectBlockGpu(cpuAttnInBuf, r1AttnHc.norm, r2Lin.branchOutBuf, wBufs.attnInject, numPos);
        const r2MlpHc = runHcForwardGpu(r2Inject.injected, wBufs.mlpNorm, wBufs.mlpDown, wBufs.mlpUp, numPos);
        const r2RouterLogits = runMatmulRowSubgroupGpu(r2MlpHc.branchInput, wBufs.routerGate, numPos, params.expertCount, params.hiddenSize);

        await postBytes('r2_attn_branchOutput.f32', await readGpuBuffer(r2Lin.branchOutBuf, numPos * params.hiddenSize * 4));
        await postBytes('r2_mlp_branchInput.f32', await readGpuBuffer(r2MlpHc.branchInput, numPos * params.hiddenSize * 4));
        await postBytes('r2_router_logits.f32', await readGpuBuffer(r2RouterLogits, numPos * params.expertCount * 4));

        await device.queue.onSubmittedWorkDone();
        return {
          adapter: {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            device: adapter.info.device,
            description: adapter.info.description,
          },
          newGpuInvocations,
          errors,
        };
      } finally {
        await device.queue.onSubmittedWorkDone();
        for (const b of owned) b.destroy();
        device.destroy();
      }
    },
    { shaders, params: p }
  );
  assert.deepEqual(gpuMeta.errors, []);
  assert(gpuMeta.newGpuInvocations > 0);
} finally {
  await browser?.close();
  server.closeAllConnections();
  await new Promise((resolve) => server.close(resolve));
}

const getOutF32 = (name) => bufferToF32(postedOutputs.get(name));

// Verify all R0 physical GPU stages against retained GPU captures AND offline JS WGSL witness
const r0StageChecks = [
  ['attn_hc.norm', getOutF32('r0_attn_norm.f32'), gpuStages['attn_hc.norm'], jsR0AttnHc.norm],
  ['attn_hc.down', getOutF32('r0_attn_down.f32'), gpuStages['attn_hc.down'], jsR0AttnHc.down],
  ['attn_hc.lowRank', getOutF32('r0_attn_lowRank.f32'), gpuStages['attn_hc.lowRank'], jsR0AttnHc.lowRank],
  ['attn_hc.up', getOutF32('r0_attn_up.f32'), gpuStages['attn_hc.up'], jsR0AttnHc.up],
  ['attn_hc.mix', getOutF32('r0_attn_mix.f32'), gpuStages['attn_hc.mix'], jsR0AttnHc.mix],
  ['attn_hc.branchInput', getOutF32('r0_attn_branchInput.f32'), gpuStages['attn_hc.branchInput'], jsR0AttnHc.branchInput],
  ['linear.attn.qkv_proj@positions', getOutF32('r0_pos_qkv.f32'), slicePosRows(gpuStages['linear.attn.qkv_proj'], p.qkvDim), jsR0Lin.posQkv],
  ['linear.attn.linear_z_proj@positions', getOutF32('r0_pos_z.f32'), slicePosRows(gpuStages['linear.attn.linear_z_proj'], p.zDim), jsR0Lin.posZ],
  ['linear.attn.linear_a_proj@positions', getOutF32('r0_pos_a.f32'), slicePosRows(gpuStages['linear.attn.linear_a_proj'], p.numValueHeads), jsR0Lin.posA],
  ['linear.attn.linear_b_proj@positions', getOutF32('r0_pos_b.f32'), slicePosRows(gpuStages['linear.attn.linear_b_proj'], p.numValueHeads), jsR0Lin.posB],
  ['linear.attn.linear_core_out_full_seq', getOutF32('r0_seq_core.f32'), gpuStages['linear.attn.linear_core_out'], jsR0Lin.seqCore],
  ['attn_hc.branchOutput', getOutF32('r0_attn_branchOutput.f32'), gpuStages['attn_hc.branchOutput'], jsR0Lin.branchOutput],
  ['attn_hc.logits', getOutF32('r0_attn_logits.f32'), gpuStages['attn_hc.logits'], jsR0Inject.logits],
  ['attn_hc.gate', getOutF32('r0_attn_gate.f32'), gpuStages['attn_hc.gate'], jsR0Inject.gate],
  ['attn_hc.injected', getOutF32('r0_attn_injected.f32'), gpuStages['attn_hc.injected'], jsR0Inject.injected],
  ['mlp_hc.norm', getOutF32('r0_mlp_norm.f32'), gpuStages['mlp_hc.norm'], jsR0MlpHc.norm],
  ['mlp_hc.down', getOutF32('r0_mlp_down.f32'), gpuStages['mlp_hc.down'], jsR0MlpHc.down],
  ['mlp_hc.lowRank', getOutF32('r0_mlp_lowRank.f32'), gpuStages['mlp_hc.lowRank'], jsR0MlpHc.lowRank],
  ['mlp_hc.up', getOutF32('r0_mlp_up.f32'), gpuStages['mlp_hc.up'], jsR0MlpHc.up],
  ['mlp_hc.mix', getOutF32('r0_mlp_mix.f32'), gpuStages['mlp_hc.mix'], jsR0MlpHc.mix],
  ['mlp_hc.branchInput', getOutF32('r0_mlp_branchInput.f32'), gpuStages['mlp_hc.branchInput'], jsR0MlpHc.branchInput],
  ['router25.logits', getOutF32('r0_router_logits.f32'), gpuStages['router25.logits'], jsR0RouterLogits],
].map(([stage, physGpu, capturedGpu, jsWitness]) => {
  const vsCaptured = compareF32Bits(physGpu, capturedGpu);
  const vsJsWitness = compareF32Bits(physGpu, jsWitness);
  return {
    stage,
    elements: physGpu.length,
    physicalGpuSha256: f32Sha(physGpu),
    vsCapturedGpu: vsCaptured,
    vsJsWitness,
  };
});

const routerWeightsByExpert = Array.from({ length: p.expertCount }, (_, e) =>
  Array.from(W.routerGate.subarray(e * p.hiddenSize, (e + 1) * p.hiddenSize))
);
const rankTopK = (values) =>
  values
    .map((value, expert) => ({ value, expert }))
    .sort((a, b) => b.value - a.value || a.expert - b.expert)
    .slice(0, p.topK)
    .map((r) => r.expert);

const cpuExactTopByPos = positions.map((_, pi) => {
  const ref = Array.from(cpuStages['mlp_hc.branchInput'].subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize));
  const exactLogits = routerWeightsByExpert.map((w) => dotEnvelope(ref, w, 0).exact);
  return rankTopK(exactLogits);
});

function evaluatePhysicalCounterfactual(id, description, physBranchInput, physRouterLogits, jsBranchInput, jsRouterLogits) {
  const branchVsJs = compareF32Bits(physBranchInput, jsBranchInput);
  const routerVsJs = compareF32Bits(physRouterLogits, jsRouterLogits);
  const perPosition = positions.map((pos, pi) => {
    const xArr = Array.from(physBranchInput.subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize));
    const exactLogits = routerWeightsByExpert.map((w) => dotEnvelope(xArr, w, 0).exact);
    const gpuLogits = Array.from(physRouterLogits.subarray(pi * p.expertCount, (pi + 1) * p.expertCount));
    const exactTop = rankTopK(exactLogits);
    const gpuTop = rankTopK(gpuLogits);
    const refTop = cpuExactTopByPos[pi];
    const exactOverlapWithCpu = exactTop.filter((e) => refTop.includes(e)).length;
    const gpuOverlapWithCpu = gpuTop.filter((e) => refTop.includes(e)).length;
    let pairMargin = null;
    if (pos === 1717) {
      pairMargin = {
        sourceExpert: 493,
        gpuExpert: 330,
        exactMargin: exactLogits[493] - exactLogits[330],
        physicalGpuMargin: gpuLogits[493] - gpuLogits[330],
        flipsExact: exactLogits[493] - exactLogits[330] < 0,
        flipsPhysicalGpu: gpuLogits[493] - gpuLogits[330] < 0,
      };
    } else if (pos === 2271) {
      pairMargin = {
        sourceExpert: 311,
        gpuExpert: 411,
        exactMargin: exactLogits[311] - exactLogits[411],
        physicalGpuMargin: gpuLogits[311] - gpuLogits[411],
        flipsExact: exactLogits[311] - exactLogits[411] < 0,
        flipsPhysicalGpu: gpuLogits[311] - gpuLogits[411] < 0,
      };
    }
    return {
      position: pos,
      exactTop,
      physicalGpuTop: gpuTop,
      exactOverlapWithCpu,
      physicalGpuOverlapWithCpu: gpuOverlapWithCpu,
      exactMatchesCpuTop10: exactOverlapWithCpu === p.topK,
      physicalGpuMatchesCpuTop10: gpuOverlapWithCpu === p.topK,
      pairMargin,
    };
  });
  return {
    id,
    description,
    physicalBranchInputSha256: f32Sha(physBranchInput),
    physicalRouterLogitsSha256: f32Sha(physRouterLogits),
    branchInputVsJsWitness: branchVsJs,
    routerLogitsVsJsWitness: routerVsJs,
    perPosition,
  };
}

const physicalCounterfactuals = [
  evaluatePhysicalCounterfactual(
    'R0_physical_gpu_all_layer25',
    'Physical WebGPU execution of Layer 25 starting from retained GPU Layer-24 stream input',
    getOutF32('r0_mlp_branchInput.f32'),
    getOutF32('r0_router_logits.f32'),
    jsR0MlpHc.branchInput,
    jsR0RouterLogits
  ),
  evaluatePhysicalCounterfactual(
    'R1_physical_gpu_cpu_stream_and_branch_into_gpu_hc',
    'Physical WebGPU execution of Layer 25 attn_hc, inject, mlp_hc, and router25 on substituted CPU attn_hc.input + CPU attn_hc.branchOutput',
    getOutF32('r1_mlp_branchInput.f32'),
    getOutF32('r1_router_logits.f32'),
    jsR1MlpHc.branchInput,
    jsR1RouterLogits
  ),
  evaluatePhysicalCounterfactual(
    'R2_physical_gpu_cpu_layer24_stream_into_all_gpu_layer25',
    'Physical WebGPU execution of all Layer-25 operators (attn_hc -> linear_attn -> inject -> mlp_hc -> router25) on substituted CPU attn_hc.input at [1717, 2271, 2303]',
    getOutF32('r2_mlp_branchInput.f32'),
    getOutF32('r2_router_logits.f32'),
    jsR2MlpHc.branchInput,
    jsR2RouterLogits
  ),
];

const r2BranchOutVsJs = compareF32Bits(getOutF32('r2_attn_branchOutput.f32'), jsR2Lin.branchOutput);

const receipt = {
  schema: 'doppler.layer25-physical-gpu-counterfactual/v1',
  qualified: false,
  executedOnPhysicalGpu: true,
  host: process.platform,
  adapter: gpuMeta.adapter,
  newGpuInvocations: gpuMeta.newGpuInvocations,
  protocolSha256: sha(protocolBytes),
  toolSha256: sha(fs.readFileSync(new URL(import.meta.url))),
  witnessCoreSha256: sha(fs.readFileSync(new URL('./layer25-wgsl-witness-core.js', import.meta.url))),
  shaderIdentities,
  identities,
  weightIdentities,
  summary: {
    newGpuInvocations: gpuMeta.newGpuInvocations,
    r0StagesChecked: r0StageChecks.length,
    r0AllMatchCapturedGpu0Ulp: r0StageChecks.every((s) => s.vsCapturedGpu.allExact),
    r0AllMatchJsWitness0Ulp: r0StageChecks.every((s) => s.vsJsWitness.allExact),
    r1PhysicalGpuMatchesJsWitness0Ulp:
      physicalCounterfactuals[1].branchInputVsJsWitness.allExact &&
      physicalCounterfactuals[1].routerLogitsVsJsWitness.allExact,
    r2PhysicalGpuMatchesJsWitness0Ulp:
      r2BranchOutVsJs.allExact &&
      physicalCounterfactuals[2].branchInputVsJsWitness.allExact &&
      physicalCounterfactuals[2].routerLogitsVsJsWitness.allExact,
    r1PhysicalGpuPreservesCpuTop10AllPositions: physicalCounterfactuals[1].perPosition.every(
      (r) => r.exactMatchesCpuTop10 && r.physicalGpuMatchesCpuTop10
    ),
    r2PhysicalGpuPreservesCpuTop10AllPositions: physicalCounterfactuals[2].perPosition.every(
      (r) => r.exactMatchesCpuTop10 && r.physicalGpuMatchesCpuTop10
    ),
  },
  r0StageChecks,
  r2BranchOutVsJsWitness: r2BranchOutVsJs,
  physicalCounterfactuals,
  scope:
    'Fresh physical WebGPU execution of Layer-25 WGSL shaders on substituted CPU inputs (R1_physical_gpu, R2_physical_gpu) and retained GPU inputs (R0_physical_gpu). Does not change Flash-Next qualification status.',
};

fs.writeFileSync(outputPath, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ adapter: receipt.adapter, summary: receipt.summary }, null, 2));
