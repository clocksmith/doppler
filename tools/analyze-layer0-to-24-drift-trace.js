#!/usr/bin/env node
// Backward drift trace and implementation-vs-variation separation across Layers 0..25
// for Qwen 3.5 Flash-Next on the 2,304-token prompt.
// Uses frozen GPU prefix stage captures (stages.json), GPU router captures, and CPU reference traces.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
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
  compareF32Bits,
} from './layer25-wgsl-witness-core.js';

const [protocolPath, artifactDir, prefixDir, qualifyDir, layer25TraceDir, routerTraceDir, outputPath] =
  process.argv.slice(2);
if (!outputPath) {
  throw new Error(
    'Usage: analyze-layer0-to-24-drift-trace.js protocol.json artifactDir prefixDir qualifyDir layer25TraceDir routerTraceDir output.json'
  );
}

const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const protocolBytes = fs.readFileSync(protocolPath);
const p = JSON.parse(protocolBytes);
if (p.schema !== 'doppler.layer0-to-24-drift-trace-protocol/v1' || p.qualificationAuthority !== false) {
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
const prefixObs = JSON.parse(
  checked(path.join(prefixDir, p.prefixStagesObservation), p.prefixStagesObservationSha256, p.prefixStagesObservation)
);
const routerObs = JSON.parse(
  checked(path.join(qualifyDir, p.routerObservation), p.routerObservationSha256, p.routerObservation)
);
const l25Trace = JSON.parse(
  checked(path.join(layer25TraceDir, p.layer25TraceReceipt), p.layer25TraceReceiptSha256, `layer25/${p.layer25TraceReceipt}`)
);
const routerTrace = JSON.parse(
  checked(path.join(routerTraceDir, p.routerTraceReceipt), p.routerTraceReceiptSha256, `router/${p.routerTraceReceipt}`)
);

assert(prefixObs.manifestSha256 === p.manifestSha256 && routerObs.manifestSha256 === p.manifestSha256);
assert(l25Trace.manifestSha256 === p.manifestSha256 && routerTrace.manifestSha256 === p.manifestSha256);

function bufferToF32(buf) {
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

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

const weightIdentities = [];
function loadWeightF32(layer, shortName) {
  const fullName = `model.language_model.layers.${layer}.${shortName}`;
  const desc = manifest.tensors[fullName];
  if (!desc) throw new Error(`Missing tensor in manifest: ${fullName}`);
  const spans = desc.spans ?? [{ shardIndex: desc.shard, offset: desc.offset, size: desc.size }];
  const raw = Buffer.concat(
    spans.map((s) => readShardRange(path.join(artifactDir, manifest.shards[s.shardIndex].filename), s.offset, s.size))
  );
  weightIdentities.push({ tensor: fullName, dtype: desc.dtype, shape: desc.shape, bytes: raw.length, sha256: sha(raw) });
  const count = raw.length / 2;
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) out[i] = halfValue(raw.readUInt16LE(i * 2));
  return out;
}

// Load all 570 GPU prefix stage tensors across Layers 0..25
const gpuStages = new Map();
for (const t of prefixObs.tensors) {
  const key = `${t.layer}:${t.role}.${t.stage}`;
  const bytes = checked(path.join(prefixDir, path.basename(t.file)), t.sha256, `prefix/${path.basename(t.file)}`);
  gpuStages.set(key, bufferToF32(bytes));
}

// Load Layer-25 CPU stage traces
const cpuL25Stages = {};
for (const entry of l25Trace.tensors) {
  const key = `${entry.role}.${entry.stage}`;
  const bytes = checked(path.join(layer25TraceDir, entry.file), entry.sha256, `layer25/${entry.file}`);
  cpuL25Stages[key] = readNpyF32(bytes).values;
}

function readNpzF32(npzPath, member) {
  return readNpyF32(execFileSync('unzip', ['-p', npzPath, `${member}.npy`], { maxBuffer: 64 * 1024 * 1024 })).values;
}

// Independent F64 reference operators for same-input local rounding vs multi-layer drift decomposition
function cpuHcNormF64(input, weight, rows) {
  const out = new Float32Array(rows * p.streamWidth);
  for (let r = 0; r < rows * p.numStreams; r++) {
    const base = r * p.hiddenSize;
    const wBase = (r % p.numStreams) * p.hiddenSize;
    let sumSq = 0;
    for (let i = 0; i < p.hiddenSize; i++) {
      const v = input[base + i];
      sumSq += v * v;
    }
    const invRms = 1 / Math.sqrt(sumSq / p.hiddenSize + p.rmsNormEps);
    for (let i = 0; i < p.hiddenSize; i++) {
      out[base + i] = Math.fround(input[base + i] * invRms * (1 + weight[wBase + i]));
    }
  }
  return out;
}

function cpuMatmulF64(input, rows, K, weight, N) {
  const out = new Float32Array(rows * N);
  for (let r = 0; r < rows; r++) {
    const xBase = r * K;
    for (let c = 0; c < N; c++) {
      const wBase = c * K;
      let sum = 0;
      for (let k = 0; k < K; k++) sum += input[xBase + k] * weight[wBase + k];
      out[r * N + c] = Math.fround(sum);
    }
  }
  return out;
}

function cpuHcForwardF64(streamIn, normW, downW, upW) {
  const norm = cpuHcNormF64(streamIn, normW, numPos);
  const down = cpuMatmulF64(norm, numPos, p.streamWidth, downW, p.lowRankDim);
  const lowRank = new Float32Array(down.length);
  for (let i = 0; i < down.length; i++) {
    const s = down[i] / p.numStreams;
    lowRank[i] = Math.fround(s / (1 + Math.exp(-s)));
  }
  const up = cpuMatmulF64(lowRank, numPos, p.lowRankDim, upW, p.streamWidth);
  const mix = new Float32Array(up.length);
  for (let i = 0; i < up.length; i++) {
    mix[i] = Math.fround(1 / (1 + Math.exp(-up[i])));
  }
  const branchInput = new Float32Array(numPos * p.hiddenSize);
  for (let r = 0; r < numPos; r++) {
    const rBase = r * p.streamWidth;
    const oBase = r * p.hiddenSize;
    for (let c = 0; c < p.hiddenSize; c++) {
      let sum = 0;
      for (let s = 0; s < p.numStreams; s++) {
        const idx = rBase + s * p.hiddenSize + c;
        sum += norm[idx] * mix[idx];
      }
      branchInput[oBase + c] = Math.fround(sum / p.numStreams);
    }
  }
  return { norm, down, lowRank, up, mix, branchInput };
}

function cpuHcInjectF64(streamIn, norm, branchOut, injectW) {
  const logits = cpuMatmulF64(norm, numPos, p.streamWidth, injectW, p.numStreams);
  const gate = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    const s = logits[i] / p.numStreams;
    gate[i] = Math.fround(2 / (1 + Math.exp(-s)));
  }
  const injected = new Float32Array(numPos * p.streamWidth);
  for (let r = 0; r < numPos; r++) {
    const rBase = r * p.streamWidth;
    const bBase = r * p.hiddenSize;
    const gBase = r * p.numStreams;
    for (let s = 0; s < p.numStreams; s++) {
      const g = gate[gBase + s];
      const sBase = rBase + s * p.hiddenSize;
      for (let c = 0; c < p.hiddenSize; c++) {
        injected[sBase + c] = Math.fround(streamIn[sBase + c] + branchOut[bBase + c] * g);
      }
    }
  }
  return { logits, gate, injected };
}

function runWgslHcForward(streamIn, normW, downW, upW) {
  const norm = witnessHyperConnectionNorm(streamIn, normW, numPos);
  const down = witnessMatmulRowSubgroupWave64(norm, numPos, p.streamWidth, downW, p.lowRankDim);
  const lowRank = witnessLowRankSilu(down, p.numStreams);
  const up = witnessMatmulRowSubgroupWave64(lowRank, numPos, p.lowRankDim, upW, p.streamWidth);
  const mix = witnessMixSigmoid(up);
  const branchInput = witnessHyperConnectionBranch(norm, mix, numPos, p.numStreams, p.hiddenSize);
  return { norm, down, lowRank, up, mix, branchInput };
}

function runWgslHcInject(streamIn, norm, branchOut, injectW) {
  const logits = witnessMatmulRowSubgroupWave64(norm, numPos, p.streamWidth, injectW, p.numStreams);
  const gate = witnessInjectGate(logits, p.numStreams);
  const injected = witnessHyperConnectionInject(streamIn, branchOut, gate, numPos, p.numStreams, p.hiddenSize);
  return { logits, gate, injected };
}

function computeRelRms(actual, reference) {
  let diffSq = 0;
  let refSq = 0;
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i] - reference[i];
    const ad = Math.abs(d);
    if (ad > maxAbs) maxAbs = ad;
    diffSq += d * d;
    refSq += reference[i] * reference[i];
  }
  const diffRms = Math.sqrt(diffSq / actual.length);
  const refRms = Math.sqrt(refSq / actual.length);
  return { diffRms, refRms, relativeRms: diffRms / refRms, maxAbs };
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Layer 25 critical decision vectors: d1717 = w493 - w330, d2271 = w311 - w411
const wGate25 = loadWeightF32(25, 'mlp.gate.weight');
const d1717 = new Float64Array(p.hiddenSize);
const d2271 = new Float64Array(p.hiddenSize);
let d1717NormSq = 0;
let d2271NormSq = 0;
for (let i = 0; i < p.hiddenSize; i++) {
  d1717[i] = wGate25[493 * p.hiddenSize + i] - wGate25[330 * p.hiddenSize + i];
  d2271[i] = wGate25[311 * p.hiddenSize + i] - wGate25[411 * p.hiddenSize + i];
  d1717NormSq += d1717[i] * d1717[i];
  d2271NormSq += d2271[i] * d2271[i];
}
const d1717Norm = Math.sqrt(d1717NormSq);
const d2271Norm = Math.sqrt(d2271NormSq);

// 1. Verify all 26 layers (L = 0..25):
//    - Constructive WGSL 0-ULP stage compliance (569 non-PLE checks)
//    - Same-input local F32 WGSL vs F64 reference rounding variation
//    - Per-layer propagated GPU vs CPU drift and decision-vector projections
//    - 59,904 token-layer rank-10 vs rank-11 margin census
let totalWgslChecks = 0;
let exact0UlpWgslChecks = 0;
let totalWgslElementsChecked = 0;
let exact0UlpWgslElements = 0;

const perLayerVerification = [];
const perLayerDriftProgression = [];
const perLayerMarginCensus = [];
let l25Weights = null;

for (let l = 0; l < p.layers; l++) {
  const attnType = (l + 1) % 4 === 0 ? 'full_attention' : 'linear_attention';
  const W = {
    attnNorm: loadWeightF32(l, 'attn_hyper_connection.hc_norm.weight'),
    attnDown: loadWeightF32(l, 'attn_hyper_connection.input_mix_weight_down.weight'),
    attnUp: loadWeightF32(l, 'attn_hyper_connection.input_mix_weight_up.weight'),
    attnInject: loadWeightF32(l, 'attn_hyper_connection.block_inject_weight.weight'),
    mlpNorm: loadWeightF32(l, 'mlp_hyper_connection.hc_norm.weight'),
    mlpDown: loadWeightF32(l, 'mlp_hyper_connection.input_mix_weight_down.weight'),
    mlpUp: loadWeightF32(l, 'mlp_hyper_connection.input_mix_weight_up.weight'),
    mlpInject: loadWeightF32(l, 'mlp_hyper_connection.block_inject_weight.weight'),
    routerGate: l === 25 ? wGate25 : loadWeightF32(l, 'mlp.gate.weight'),
  };
  if (l === 25) l25Weights = W;

  const g = (stage) => gpuStages.get(`${l}:${stage}`);
  const rRec = routerObs.rows[0].routerObservations.find((r) => r.layer === l);
  const gpuRouterInAll = bufferToF32(
    checked(path.join(qualifyDir, path.basename(rRec.input.file)), rRec.input.sha256, `router_gpu/${path.basename(rRec.input.file)}`)
  );
  const gpuRouterLogitsAll = bufferToF32(
    checked(path.join(qualifyDir, path.basename(rRec.logits.file)), rRec.logits.sha256, `router_gpu/${path.basename(rRec.logits.file)}`)
  );
  const gpuRouterInPos = slicePosRows(gpuRouterInAll, p.hiddenSize);
  const gpuRouterLogitsPos = slicePosRows(gpuRouterLogitsAll, p.expertCount);

  const routerNpzName = `router-${l}.npz`;
  const routerNpzPath = path.join(routerTraceDir, routerNpzName);
  checked(routerNpzPath, routerTrace.files[routerNpzName], `router_cpu/${routerNpzName}`);
  const cpuRouterInAll = readNpzF32(routerNpzPath, 'input');
  const cpuRouterLogitsAll = readNpzF32(routerNpzPath, 'logits');
  const cpuRouterInPos = slicePosRows(cpuRouterInAll, p.hiddenSize);

  const checks = [
    ['attn_hc.norm', witnessHyperConnectionNorm(g('attn_hc.input'), W.attnNorm, numPos), g('attn_hc.norm')],
    ['attn_hc.down', witnessMatmulRowSubgroupWave64(g('attn_hc.norm'), numPos, p.streamWidth, W.attnDown, p.lowRankDim), g('attn_hc.down')],
    ['attn_hc.lowRank', witnessLowRankSilu(g('attn_hc.down'), p.numStreams), g('attn_hc.lowRank')],
    ['attn_hc.up', witnessMatmulRowSubgroupWave64(g('attn_hc.lowRank'), numPos, p.lowRankDim, W.attnUp, p.streamWidth), g('attn_hc.up')],
    ['attn_hc.mix', witnessMixSigmoid(g('attn_hc.up')), g('attn_hc.mix')],
    ['attn_hc.branchInput', witnessHyperConnectionBranch(g('attn_hc.norm'), g('attn_hc.mix'), numPos, p.numStreams, p.hiddenSize), g('attn_hc.branchInput')],
    ['attn_hc.logits', witnessMatmulRowSubgroupWave64(g('attn_hc.norm'), numPos, p.streamWidth, W.attnInject, p.numStreams), g('attn_hc.logits')],
    ['attn_hc.gate', witnessInjectGate(g('attn_hc.logits'), p.numStreams), g('attn_hc.gate')],
    ['attn_hc.injected', witnessHyperConnectionInject(g('attn_hc.input'), g('attn_hc.branchOutput'), g('attn_hc.gate'), numPos, p.numStreams, p.hiddenSize), g('attn_hc.injected')],
    ['mlp_hc.input==attn_hc.injected', g('attn_hc.injected'), g('mlp_hc.input')],
    ['mlp_hc.norm', witnessHyperConnectionNorm(g('mlp_hc.input'), W.mlpNorm, numPos), g('mlp_hc.norm')],
    ['mlp_hc.down', witnessMatmulRowSubgroupWave64(g('mlp_hc.norm'), numPos, p.streamWidth, W.mlpDown, p.lowRankDim), g('mlp_hc.down')],
    ['mlp_hc.lowRank', witnessLowRankSilu(g('mlp_hc.down'), p.numStreams), g('mlp_hc.lowRank')],
    ['mlp_hc.up', witnessMatmulRowSubgroupWave64(g('mlp_hc.lowRank'), numPos, p.lowRankDim, W.mlpUp, p.streamWidth), g('mlp_hc.up')],
    ['mlp_hc.mix', witnessMixSigmoid(g('mlp_hc.up')), g('mlp_hc.mix')],
    ['mlp_hc.branchInput', witnessHyperConnectionBranch(g('mlp_hc.norm'), g('mlp_hc.mix'), numPos, p.numStreams, p.hiddenSize), g('mlp_hc.branchInput')],
    ['mlp_hc.logits', witnessMatmulRowSubgroupWave64(g('mlp_hc.norm'), numPos, p.streamWidth, W.mlpInject, p.numStreams), g('mlp_hc.logits')],
    ['mlp_hc.gate', witnessInjectGate(g('mlp_hc.logits'), p.numStreams), g('mlp_hc.gate')],
    ['router.input==mlp_hc.branchInput', g('mlp_hc.branchInput'), gpuRouterInPos],
    ['router.logits', witnessMatmulRowSubgroupWave64(gpuRouterInPos, numPos, p.hiddenSize, W.routerGate, p.expertCount), gpuRouterLogitsPos],
  ];
  if (g('mlp_hc.injected')) {
    checks.push([
      'mlp_hc.injected',
      witnessHyperConnectionInject(g('mlp_hc.input'), g('mlp_hc.branchOutput'), g('mlp_hc.gate'), numPos, p.numStreams, p.hiddenSize),
      g('mlp_hc.injected'),
    ]);
  }
  if (l >= 2) {
    checks.push([
      'attn_hc.input==prev_layer.mlp_hc.injected',
      gpuStages.get(`${l - 1}:mlp_hc.injected`),
      g('attn_hc.input'),
    ]);
  }

  let layerChecksExact = 0;
  for (const [name, pred, actual] of checks) {
    const cmp = compareF32Bits(pred, actual);
    if (!cmp.allExact) {
      throw new Error(`Layer ${l} stage ${name} failed 0-ULP WGSL witness check: ${JSON.stringify(cmp)}`);
    }
    totalWgslChecks += 1;
    exact0UlpWgslChecks += 1;
    layerChecksExact += 1;
    totalWgslElementsChecked += cmp.total;
    exact0UlpWgslElements += cmp.exact;
  }

  // Same-input local F32 WGSL vs F64 reference variation on identical GPU inputs
  const localAttnF64 = cpuHcForwardF64(g('attn_hc.input'), W.attnNorm, W.attnDown, W.attnUp);
  const localAttnInjectF64 = cpuHcInjectF64(g('attn_hc.input'), g('attn_hc.norm'), g('attn_hc.branchOutput'), W.attnInject);
  const localMlpF64 = cpuHcForwardF64(g('mlp_hc.input'), W.mlpNorm, W.mlpDown, W.mlpUp);
  const localRouterF64 = cpuMatmulF64(gpuRouterInPos, numPos, p.hiddenSize, W.routerGate, p.expertCount);

  const attnHcLocalRel = computeRelRms(g('attn_hc.branchInput'), localAttnF64.branchInput);
  const attnInjectLocalRel = computeRelRms(g('attn_hc.injected'), localAttnInjectF64.injected);
  const mlpHcLocalRel = computeRelRms(g('mlp_hc.branchInput'), localMlpF64.branchInput);
  const routerLocalRel = computeRelRms(gpuRouterLogitsPos, localRouterF64);

  perLayerVerification.push({
    layer: l,
    attentionType: attnType,
    hasPleAdditionOnEntry: l === 1,
    wgslChecksPassed0Ulp: layerChecksExact,
    wgslChecksTotal: checks.length,
    sameInputLocalF32VsF64: {
      attnHcBranchInputRelRms: attnHcLocalRel.relativeRms,
      attnHcBranchInputMaxUlp: compareF32Bits(g('attn_hc.branchInput'), localAttnF64.branchInput).maxUlp,
      attnInjectRelRms: attnInjectLocalRel.relativeRms,
      attnInjectMaxUlp: compareF32Bits(g('attn_hc.injected'), localAttnInjectF64.injected).maxUlp,
      mlpHcBranchInputRelRms: mlpHcLocalRel.relativeRms,
      mlpHcBranchInputMaxUlp: compareF32Bits(g('mlp_hc.branchInput'), localMlpF64.branchInput).maxUlp,
      routerLogitsRelRms: routerLocalRel.relativeRms,
      routerLogitsMaxUlp: compareF32Bits(gpuRouterLogitsPos, localRouterF64).maxUlp,
    },
  });

  // Propagated GPU vs CPU drift on mlp_hc.branchInput (router.input)
  const fullSeqDrift = computeRelRms(gpuRouterInAll, cpuRouterInAll);
  const posMetrics = positions.map((pos, pi) => {
    const gRow = g('mlp_hc.branchInput').subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize);
    const cRow = cpuRouterInPos.subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize);
    const rel = computeRelRms(gRow, cRow);
    const diffVec = new Float64Array(p.hiddenSize);
    for (let i = 0; i < p.hiddenSize; i++) diffVec[i] = gRow[i] - cRow[i];
    return {
      position: pos,
      diffRms: rel.diffRms,
      sourceRms: rel.refRms,
      relativeRms: rel.relativeRms,
      maxAbs: rel.maxAbs,
      projOnL25D1717: pos === 1717 ? dot(diffVec, d1717) : undefined,
      projOnL25D2271: pos === 2271 ? dot(diffVec, d2271) : undefined,
    };
  });

  perLayerDriftProgression.push({
    layer: l,
    attentionType: attnType,
    fullSeqRouterInRelRms: fullSeqDrift.relativeRms,
    positions: posMetrics,
  });

  // Token-layer rank-10 vs rank-11 margin census across all 2,304 tokens
  let minMargin = Infinity;
  let minMarginToken = -1;
  let countLe1e6 = 0;
  let countLe2e6 = 0;
  let mismatchTokens = 0;
  for (let t = 0; t < p.seqLen; t++) {
    const cBase = t * p.expertCount;
    const gBase = t * p.expertCount;
    const cIdx = Array.from({ length: p.expertCount }, (_, i) => i).sort(
      (a, b) => cpuRouterLogitsAll[cBase + b] - cpuRouterLogitsAll[cBase + a] || a - b
    );
    const gIdx = Array.from({ length: p.expertCount }, (_, i) => i).sort(
      (a, b) => gpuRouterLogitsAll[gBase + b] - gpuRouterLogitsAll[gBase + a] || a - b
    );
    const cTop10 = new Set(cIdx.slice(0, p.topK));
    let overlap = 0;
    for (let k = 0; k < p.topK; k++) if (cTop10.has(gIdx[k])) overlap++;
    if (overlap < p.topK) mismatchTokens++;
    const margin = cpuRouterLogitsAll[cBase + cIdx[9]] - cpuRouterLogitsAll[cBase + cIdx[10]];
    if (margin <= 1e-6) countLe1e6++;
    if (margin <= 2e-6) countLe2e6++;
    if (margin < minMargin) {
      minMargin = margin;
      minMarginToken = t;
    }
  }
  const margin1717 = (() => {
    const cBase = 1717 * p.expertCount;
    const cIdx = Array.from({ length: p.expertCount }, (_, i) => i).sort(
      (a, b) => cpuRouterLogitsAll[cBase + b] - cpuRouterLogitsAll[cBase + a] || a - b
    );
    return cpuRouterLogitsAll[cBase + cIdx[9]] - cpuRouterLogitsAll[cBase + cIdx[10]];
  })();
  const margin2271 = (() => {
    const cBase = 2271 * p.expertCount;
    const cIdx = Array.from({ length: p.expertCount }, (_, i) => i).sort(
      (a, b) => cpuRouterLogitsAll[cBase + b] - cpuRouterLogitsAll[cBase + a] || a - b
    );
    return cpuRouterLogitsAll[cBase + cIdx[9]] - cpuRouterLogitsAll[cBase + cIdx[10]];
  })();
  perLayerMarginCensus.push({
    layer: l,
    attentionType: attnType,
    tokensChecked: p.seqLen,
    mismatchTokens,
    countMarginLe1e6: countLe1e6,
    countMarginLe2e6: countLe2e6,
    minMargin,
    minMarginToken,
    marginAt1717: margin1717,
    marginAt2271: margin2271,
  });
}

// 2. Exact 4-Stream Checkpoint Attribution across C in [0, 3, 4, 8, 12, 16, 20, 24, 25]
const streamDeltaByCheckpoint = new Map();
const streamCheckpointMetrics = [];
for (const c of p.streamCheckpoints) {
  const gpuPos = gpuStages.get(`${c}:attn_hc.input`);
  let cpuPos;
  if (c === 0) {
    // Layer 0 attn_hc.input is the replicated F16 embedding table (identical on CPU and GPU).
    cpuPos = new Float32Array(gpuPos);
  } else if (c === 25) {
    cpuPos = cpuL25Stages['attn_hc.input'];
  } else {
    const npyName = `layer-${c - 1}.npy`;
    const npyPath = path.join(routerTraceDir, npyName);
    const bytes = checked(npyPath, routerTrace.files[npyName], `stream_cpu/${npyName}`);
    const cpuFull = readNpyF32(bytes).values;
    cpuPos = slicePosRows(cpuFull, p.streamWidth);
  }
  const delta = new Float32Array(numPos * p.streamWidth);
  for (let i = 0; i < delta.length; i++) delta[i] = gpuPos[i] - cpuPos[i];
  streamDeltaByCheckpoint.set(c, delta);
  const rel = computeRelRms(gpuPos, cpuPos);
  streamCheckpointMetrics.push({
    checkpointLayer: c,
    diffRms: rel.diffRms,
    sourceRms: rel.refRms,
    relativeRms: rel.relativeRms,
    maxAbs: rel.maxAbs,
  });
}

const routerWeightsByExpert25 = Array.from({ length: p.expertCount }, (_, e) =>
  Array.from(wGate25.subarray(e * p.hiddenSize, (e + 1) * p.hiddenSize))
);
const rankTopK = (values) =>
  values
    .map((value, expert) => ({ value, expert }))
    .sort((a, b) => b.value - a.value || a.expert - b.expert)
    .slice(0, p.topK)
    .map((r) => r.expert);

const cpuExactTopByPos25 = positions.map((_, pi) => {
  const ref = Array.from(cpuL25Stages['mlp_hc.branchInput'].subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize));
  const exactLogits = routerWeightsByExpert25.map((w) => dotEnvelope(ref, w, 0).exact);
  return rankTopK(exactLogits);
});

function evaluateL25WithModifiedStreams(id, description, modStreamsPos, useGpuBranchOut = false) {
  const attnHc = runWgslHcForward(modStreamsPos, l25Weights.attnNorm, l25Weights.attnDown, l25Weights.attnUp);
  const branchOut = useGpuBranchOut ? gpuStages.get('25:attn_hc.branchOutput') : cpuL25Stages['attn_hc.branchOutput'];
  const inject = runWgslHcInject(modStreamsPos, attnHc.norm, branchOut, l25Weights.attnInject);
  const mlpHc = runWgslHcForward(inject.injected, l25Weights.mlpNorm, l25Weights.mlpDown, l25Weights.mlpUp);
  const wgslLogitsAll = witnessMatmulRowSubgroupWave64(
    mlpHc.branchInput,
    numPos,
    p.hiddenSize,
    wGate25,
    p.expertCount
  );

  const perPosition = positions.map((pos, pi) => {
    const xArr = Array.from(mlpHc.branchInput.subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize));
    const exactLogits = routerWeightsByExpert25.map((w) => dotEnvelope(xArr, w, 0).exact);
    const wgslLogits = Array.from(wgslLogitsAll.subarray(pi * p.expertCount, (pi + 1) * p.expertCount));
    const exactTop = rankTopK(exactLogits);
    const wgslTop = rankTopK(wgslLogits);
    const refTop = cpuExactTopByPos25[pi];
    const exactOverlap = exactTop.filter((e) => refTop.includes(e)).length;
    const wgslOverlap = wgslTop.filter((e) => refTop.includes(e)).length;
    let pairMargin = null;
    if (pos === 1717) {
      pairMargin = {
        sourceExpert: 493,
        gpuExpert: 330,
        exactMargin: exactLogits[493] - exactLogits[330],
        wgslMargin: wgslLogits[493] - wgslLogits[330],
        flipsExact: exactLogits[493] - exactLogits[330] < 0,
        flipsWgsl: wgslLogits[493] - wgslLogits[330] < 0,
      };
    } else if (pos === 2271) {
      pairMargin = {
        sourceExpert: 311,
        gpuExpert: 411,
        exactMargin: exactLogits[311] - exactLogits[411],
        wgslMargin: wgslLogits[311] - wgslLogits[411],
        flipsExact: exactLogits[311] - exactLogits[411] < 0,
        flipsWgsl: wgslLogits[311] - wgslLogits[411] < 0,
      };
    }
    return {
      position: pos,
      exactOverlapWithCpu: exactOverlap,
      wgslOverlapWithCpu: wgslOverlap,
      exactMatchesCpuTop10: exactOverlap === p.topK,
      wgslMatchesCpuTop10: wgslOverlap === p.topK,
      pairMargin,
    };
  });

  return { id, description, useGpuBranchOut, perPosition };
}

const cpuS25 = cpuL25Stages['attn_hc.input'];
const ds25 = streamDeltaByCheckpoint.get(25);

// 2a. Cumulative Prefix Grafts: cpu_streams_25 + delta_streams_C
const cumulativePrefixGrafts = p.streamCheckpoints.map((c) => {
  const dsC = streamDeltaByCheckpoint.get(c);
  const mod = new Float32Array(cpuS25.length);
  for (let i = 0; i < mod.length; i++) mod[i] = Math.fround(cpuS25[i] + dsC[i]);
  return {
    checkpointLayer: c,
    withCpuL25BranchOut: evaluateL25WithModifiedStreams(
      `prefix_C${c}_cpuBranch`,
      `Graft cumulative 4-stream drift at layer ${c} onto CPU Layer-25 streams (with CPU Layer-25 branchOutput)`,
      mod,
      false
    ),
    withGpuL25BranchOut: evaluateL25WithModifiedStreams(
      `prefix_C${c}_gpuBranch`,
      `Graft cumulative 4-stream drift at layer ${c} onto CPU Layer-25 streams (with GPU Layer-25 branchOutput)`,
      mod,
      true
    ),
  };
});

// 2b. Per-Interval Single-Block Contribution & Leave-One-Interval-Out
const intervals = [];
for (let i = 0; i < p.streamCheckpoints.length - 1; i++) {
  intervals.push([p.streamCheckpoints[i], p.streamCheckpoints[i + 1]]);
}

const singleIntervalGrafts = [];
const leaveOneIntervalOutGrafts = [];

for (const [c0, c1] of intervals) {
  const ds0 = streamDeltaByCheckpoint.get(c0);
  const ds1 = streamDeltaByCheckpoint.get(c1);
  const inc = new Float32Array(cpuS25.length);
  const onlyInc = new Float32Array(cpuS25.length);
  const leaveOut = new Float32Array(cpuS25.length);
  let incSq = 0;
  for (let i = 0; i < inc.length; i++) {
    inc[i] = ds1[i] - ds0[i];
    incSq += inc[i] * inc[i];
    onlyInc[i] = Math.fround(cpuS25[i] + inc[i]);
    leaveOut[i] = Math.fround(cpuS25[i] + (ds25[i] - inc[i]));
  }
  const incRms = Math.sqrt(incSq / inc.length);
  const label = `L${c0}..L${c1 - 1}`;
  singleIntervalGrafts.push({
    interval: label,
    startLayer: c0,
    endLayerExclusive: c1,
    streamIncrementRms: incRms,
    withCpuL25BranchOut: evaluateL25WithModifiedStreams(
      `only_${label}_cpuBranch`,
      `Add only 4-stream drift increment from ${label} to CPU Layer-25 streams`,
      onlyInc,
      false
    ),
  });
  leaveOneIntervalOutGrafts.push({
    intervalRemoved: label,
    startLayer: c0,
    endLayerExclusive: c1,
    streamIncrementRms: incRms,
    withCpuL25BranchOut: evaluateL25WithModifiedStreams(
      `drop_${label}_cpuBranch`,
      `Full Layer-25 GPU stream drift minus ${label} increment (with CPU Layer-25 branchOutput)`,
      leaveOut,
      false
    ),
    withGpuL25BranchOut: evaluateL25WithModifiedStreams(
      `drop_${label}_gpuBranch`,
      `Full Layer-25 GPU stream drift minus ${label} increment (with GPU Layer-25 branchOutput)`,
      leaveOut,
      true
    ),
  });
}

// 2c. Suffix-Only Attribution: zero all drift before checkpoint C (keep only ds25 - dsC)
const suffixOnlyGrafts = p.streamCheckpoints.map((c) => {
  const dsC = streamDeltaByCheckpoint.get(c);
  const mod = new Float32Array(cpuS25.length);
  for (let i = 0; i < mod.length; i++) mod[i] = Math.fround(cpuS25[i] + (ds25[i] - dsC[i]));
  return {
    zeroedBeforeLayer: c,
    keptSuffix: c < 25 ? `L${c}..L24` : 'none',
    withCpuL25BranchOut: evaluateL25WithModifiedStreams(
      `suffix_from_L${c}_cpuBranch`,
      `Zero all 4-stream drift before L${c}; keep only L${c}..L24 stream drift (with CPU Layer-25 branchOutput)`,
      mod,
      false
    ),
    withGpuL25BranchOut: evaluateL25WithModifiedStreams(
      `suffix_from_L${c}_gpuBranch`,
      `Zero all 4-stream drift before L${c}; keep only L${c}..L24 stream drift (with GPU Layer-25 branchOutput)`,
      mod,
      true
    ),
  };
});

// 3. Isotropic dot-product standard deviation comparison at Layer 25
const l25Drift1717 = perLayerDriftProgression[25].positions[0];
const l25Drift2271 = perLayerDriftProgression[25].positions[1];
const sigmaDot1717 = l25Drift1717.diffRms * d1717Norm;
const sigmaDot2271 = l25Drift2271.diffRms * d2271Norm;

const dropL20To23 = leaveOneIntervalOutGrafts.find((g) => g.intervalRemoved === 'L20..L23');
const suffixFromL20 = suffixOnlyGrafts.find((g) => g.zeroedBeforeLayer === 20);
const suffixFromL24 = suffixOnlyGrafts.find((g) => g.zeroedBeforeLayer === 24);

const result = {
  schema: 'doppler.layer0-to-24-drift-trace/v1',
  qualified: false,
  protocolSha256: sha(protocolBytes),
  toolSha256: sha(fs.readFileSync(new URL(import.meta.url))),
  witnessCoreSha256: sha(fs.readFileSync(new URL('./layer25-wgsl-witness-core.js', import.meta.url))),
  arithmeticCoreSha256: sha(fs.readFileSync(new URL('./numerical-discontinuity-core.js', import.meta.url))),
  identities,
  weightIdentitiesCount: weightIdentities.length,
  weightIdentitiesSha256: sha(Buffer.from(JSON.stringify(weightIdentities))),
  summary: {
    layersVerified: p.layers,
    totalWgslChecks,
    exact0UlpWgslChecks,
    all569WgslChecksPass0Ulp: totalWgslChecks === 569 && exact0UlpWgslChecks === 569,
    totalWgslElementsChecked,
    exact0UlpWgslElements,
    maxLocalSingleBlockHcRelRms: Math.max(
      ...perLayerVerification.map((v) =>
        Math.max(v.sameInputLocalF32VsF64.attnHcBranchInputRelRms, v.sameInputLocalF32VsF64.mlpHcBranchInputRelRms)
      )
    ),
    l0FullSeqRouterInRelRms: perLayerDriftProgression[0].fullSeqRouterInRelRms,
    l25FullSeqRouterInRelRms: perLayerDriftProgression[25].fullSeqRouterInRelRms,
    totalTokenLayerDecisionsChecked: p.layers * p.seqLen,
    tokensWithMarginLe2e6InLayers13To24: perLayerMarginCensus
      .slice(13, 25)
      .reduce((acc, row) => acc + row.countMarginLe2e6, 0),
    tokensWithMarginLe2e6InLayer25: perLayerMarginCensus[25].countMarginLe2e6,
    layer25CriticalMarginsSmallerThanOneSigmaDot:
      perLayerMarginCensus[25].marginAt1717 < sigmaDot1717 && perLayerMarginCensus[25].marginAt2271 < sigmaDot2271,
    dominantInterval: 'L20..L23',
    removingOnlyL20To23RestoresBothPositionsOnGpuLayer25:
      dropL20To23.withGpuL25BranchOut.perPosition.every((r) => r.exactMatchesCpuTop10 && r.wgslMatchesCpuTop10),
    suffixL20To25AloneFlipsBothPositions:
      suffixFromL20.withGpuL25BranchOut.perPosition[0].pairMargin.flipsExact &&
      suffixFromL20.withGpuL25BranchOut.perPosition[1].pairMargin.flipsExact,
    suffixL24To25AloneFlipsNeitherPosition:
      suffixFromL24.withGpuL25BranchOut.perPosition.every((r) => r.exactMatchesCpuTop10 && r.wgslMatchesCpuTop10),
  },
  isotropicSensitivityAtLayer25: {
    d1717Norm,
    d2271Norm,
    pos1717BranchInputDiffRms: l25Drift1717.diffRms,
    pos2271BranchInputDiffRms: l25Drift2271.diffRms,
    expectedDotStd1717: sigmaDot1717,
    expectedDotStd2271: sigmaDot2271,
    cpuMargin1717: perLayerMarginCensus[25].marginAt1717,
    cpuMargin2271: perLayerMarginCensus[25].marginAt2271,
    observedDotShift1717: l25Drift1717.projOnL25D1717,
    observedDotShift2271: l25Drift2271.projOnL25D2271,
    observedZScore1717: l25Drift1717.projOnL25D1717 / sigmaDot1717,
    observedZScore2271: l25Drift2271.projOnL25D2271 / sigmaDot2271,
  },
  perLayerVerification,
  perLayerDriftProgression,
  perLayerMarginCensus,
  streamCheckpointMetrics,
  cumulativePrefixGrafts,
  singleIntervalGrafts,
  leaveOneIntervalOutGrafts,
  suffixOnlyGrafts,
  scope:
    '26-layer (L=0..25) constructive WGSL verification (569/569 checks at 0 ULP), same-input F32-vs-F64 local rounding decomposition, 59,904 token-layer margin census, and 4-stream checkpoint interval attribution. Does not qualify the model.',
};

fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result.summary, null, 2));
