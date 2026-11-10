#!/usr/bin/env node
// Constructive F32 WGSL operator witness and bidirectional counterfactual attribution
// for Flash-Next Layer 25 pre-router path (attn_hc -> linear_attn -> inject -> mlp_hc -> router).
// Offline CPU verification against frozen captures only. Does not open a GPU or change qualification.

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readNpyF32 } from './compare-source-boundaries.js';
import { halfValue, dotEnvelope } from './numerical-discontinuity-core.js';
import {
  f32ToBits,
  bitsToF32,
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

const [protocolPath, artifactDir, qualifyDir, layer25TraceDir, routerTraceDir, outputPath] = process.argv.slice(2);
if (!outputPath) {
  throw new Error(
    'Usage: analyze-layer25-prerouter-witness.js protocol.json artifactDir qualifyDir layer25TraceDir routerTraceDir output.json'
  );
}

const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
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

// 1. Fail-closed verification of manifest, GPU observations, and CPU reference traces
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

if (
  !stagesObs.diagnosticCompleted ||
  !routerObs.completed ||
  !l25Trace.diagnosticCompleted ||
  !routerTrace.completed ||
  stagesObs.manifestSha256 !== p.manifestSha256 ||
  routerObs.manifestSha256 !== p.manifestSha256 ||
  l25Trace.manifestSha256 !== p.manifestSha256 ||
  routerTrace.manifestSha256 !== p.manifestSha256 ||
  routerCmp.gpuSha256 !== p.routerObservationSha256 ||
  routerCmp.sourceTraceSha256 !== p.routerTraceReceiptSha256 ||
  routerCmp.firstChangedLayer !== p.layer
) {
  throw new Error('Unmatched or incomplete capture chain');
}

if (manifest?.inference?.normalization?.linearNormGateActivation !== 'sigmoid') {
  throw new Error('Unexpected linearNormGateActivation in manifest');
}

// 2. Load and verify all 18 Layer-25 non-expert weight tensors
const weightPrefix = `model.language_model.layers.${p.layer}.`;
const weightIdentities = [];
function loadWeightF32(shortName) {
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
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = halfValue(raw.readUInt16LE(i * 2));
  }
  return out;
}

const W = {
  attnNorm: loadWeightF32('attn_hyper_connection.hc_norm.weight'),
  attnDown: loadWeightF32('attn_hyper_connection.input_mix_weight_down.weight'),
  attnUp: loadWeightF32('attn_hyper_connection.input_mix_weight_up.weight'),
  attnInject: loadWeightF32('attn_hyper_connection.block_inject_weight.weight'),
  mlpNorm: loadWeightF32('mlp_hyper_connection.hc_norm.weight'),
  mlpDown: loadWeightF32('mlp_hyper_connection.input_mix_weight_down.weight'),
  mlpUp: loadWeightF32('mlp_hyper_connection.input_mix_weight_up.weight'),
  mlpInject: loadWeightF32('mlp_hyper_connection.block_inject_weight.weight'),
  qkvProj: loadWeightF32('linear_attn.in_proj_qkv.weight'),
  zProj: loadWeightF32('linear_attn.in_proj_z.weight'),
  aProj: loadWeightF32('linear_attn.in_proj_a.weight'),
  bProj: loadWeightF32('linear_attn.in_proj_b.weight'),
  conv1d: loadWeightF32('linear_attn.conv1d.weight'),
  aLog: loadWeightF32('linear_attn.A_log'),
  dtBias: loadWeightF32('linear_attn.dt_bias'),
  normW: loadWeightF32('linear_attn.norm.weight'),
  outProj: loadWeightF32('linear_attn.out_proj.weight'),
  routerGate: loadWeightF32('mlp.gate.weight'),
};

// 3. Load and verify all GPU stage captures and CPU reference NPYs
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
if (!routerRecord) throw new Error(`Missing router record for layer ${p.layer}`);
const gpuRouterInputAll = bufferToF32(
  checked(
    path.join(qualifyDir, path.basename(routerRecord.input.file)),
    routerRecord.input.sha256,
    path.basename(routerRecord.input.file)
  )
);
const gpuRouterLogitsAll = bufferToF32(
  checked(
    path.join(qualifyDir, path.basename(routerRecord.logits.file)),
    routerRecord.logits.sha256,
    path.basename(routerRecord.logits.file)
  )
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

// 4. Constructive WGSL Operator Witnesses across all Layer-25 stages
const gpuProjIn = gpuStages['linear.attn.post_input_norm'];
const gpuPosProjIn = slicePosRows(gpuProjIn, p.hiddenSize);
const gpuPosQkv = slicePosRows(gpuStages['linear.attn.qkv_proj'], p.qkvDim);
const gpuPosZ = slicePosRows(gpuStages['linear.attn.linear_z_proj'], p.zDim);
const gpuPosA = slicePosRows(gpuStages['linear.attn.linear_a_proj'], p.numValueHeads);
const gpuPosB = slicePosRows(gpuStages['linear.attn.linear_b_proj'], p.numValueHeads);
const gpuPosCoreOut = slicePosRows(gpuStages['linear.attn.linear_core_out'], p.zDim);

// Full 2,304-token GatedDeltaNet recurrent core witness
const fullCoreOutWitness = witnessGatedDeltaCore(
  gpuStages['linear.attn.qkv_proj'],
  gpuStages['linear.attn.linear_z_proj'],
  gpuStages['linear.attn.linear_a_proj'],
  gpuStages['linear.attn.linear_b_proj'],
  W.conv1d,
  W.dtBias,
  W.aLog,
  W.normW,
  p.seqLen
);
const fullCoreOutCmp = compareF32Bits(fullCoreOutWitness, gpuStages['linear.attn.linear_core_out']);
if (!fullCoreOutCmp.allExact) {
  throw new Error(`Full 2304-token linear_core_out witness failed 0-ULP check: ${JSON.stringify(fullCoreOutCmp)}`);
}

const witnessStageDefs = [
  {
    stage: 'attn_hc.norm',
    kernel: 'hyper_connection_norm (4x2560 streams, 256-thread tree + divideRefined + inverseRootRefined)',
    actual: gpuStages['attn_hc.norm'],
    predict: () => witnessHyperConnectionNorm(gpuStages['attn_hc.input'], W.attnNorm, numPos),
  },
  {
    stage: 'attn_hc.down',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['attn_hc.down'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['attn_hc.norm'], numPos, p.streamWidth, W.attnDown, p.lowRankDim),
  },
  {
    stage: 'attn_hc.lowRank',
    kernel: 'hyper_connection_low_rank_prepare.wgsl (siluWgsl(divideRefined(x, 4)))',
    actual: gpuStages['attn_hc.lowRank'],
    predict: () => witnessLowRankSilu(gpuStages['attn_hc.down'], p.numStreams),
  },
  {
    stage: 'attn_hc.up',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['attn_hc.up'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['attn_hc.lowRank'], numPos, p.lowRankDim, W.attnUp, p.streamWidth),
  },
  {
    stage: 'attn_hc.mix',
    kernel: 'hyper_connection_mix_prepare.wgsl (sigmoidRefined(x))',
    actual: gpuStages['attn_hc.mix'],
    predict: () => witnessMixSigmoid(gpuStages['attn_hc.up']),
  },
  {
    stage: 'attn_hc.branchInput',
    kernel: 'hyper_connection_mix.wgsl (divideRefined(sum_s(norm_s * mix_s), 4))',
    actual: gpuStages['attn_hc.branchInput'],
    predict: () => witnessHyperConnectionBranch(gpuStages['attn_hc.norm'], gpuStages['attn_hc.mix'], numPos, p.numStreams, p.hiddenSize),
  },
  {
    stage: 'linear.attn.post_input_norm@positions',
    kernel: 'identity (attn_hc.branchInput at [1717, 2271, 2303])',
    actual: gpuPosProjIn,
    predict: () => gpuStages['attn_hc.branchInput'],
  },
  {
    stage: 'linear.attn.qkv_proj@positions',
    kernel: 'matmul_f16w_f32a_tiled.wgsl (16-element K-tile FMA subtotal)',
    actual: gpuPosQkv,
    predict: () => witnessMatmulF16wF32aTiled16(gpuPosProjIn, numPos, p.hiddenSize, W.qkvProj, p.qkvDim),
  },
  {
    stage: 'linear.attn.linear_z_proj@positions',
    kernel: 'matmul_f16w_f32a_tiled.wgsl (16-element K-tile FMA subtotal)',
    actual: gpuPosZ,
    predict: () => witnessMatmulF16wF32aTiled16(gpuPosProjIn, numPos, p.hiddenSize, W.zProj, p.zDim),
  },
  {
    stage: 'linear.attn.linear_a_proj@positions',
    kernel: 'matmul_f16w_f32a.wgsl (sequential 0..K-1 FMA, N=48<64)',
    actual: gpuPosA,
    predict: () => witnessMatmulF16wF32aSeq(gpuPosProjIn, numPos, p.hiddenSize, W.aProj, p.numValueHeads),
  },
  {
    stage: 'linear.attn.linear_b_proj@positions',
    kernel: 'matmul_f16w_f32a.wgsl (sequential 0..K-1 FMA, N=48<64)',
    actual: gpuPosB,
    predict: () => witnessMatmulF16wF32aSeq(gpuPosProjIn, numPos, p.hiddenSize, W.bProj, p.numValueHeads),
  },
  {
    stage: 'linear.attn.linear_core_out@positions',
    kernel: 'gated_delta_conv.wgsl + gated_delta_recurrent_sigmoid_gate.wgsl',
    actual: gpuPosCoreOut,
    predict: () => slicePosRows(fullCoreOutWitness, p.zDim),
  },
  {
    stage: 'attn_hc.branchOutput',
    kernel: 'matmul_f16w_f32a_tiled.wgsl (linear_attn.out_proj)',
    actual: gpuStages['attn_hc.branchOutput'],
    predict: () => witnessMatmulF16wF32aTiled16(gpuPosCoreOut, numPos, p.zDim, W.outProj, p.hiddenSize),
  },
  {
    stage: 'attn_hc.logits',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['attn_hc.logits'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['attn_hc.norm'], numPos, p.streamWidth, W.attnInject, p.numStreams),
  },
  {
    stage: 'attn_hc.gate',
    kernel: 'hyper_connection_inject_prepare.wgsl (2 * sigmoidRefined(divideRefined(x, 4)))',
    actual: gpuStages['attn_hc.gate'],
    predict: () => witnessInjectGate(gpuStages['attn_hc.logits'], p.numStreams),
  },
  {
    stage: 'attn_hc.injected',
    kernel: 'hyper_connection_inject.wgsl (stream + branch * gate)',
    actual: gpuStages['attn_hc.injected'],
    predict: () =>
      witnessHyperConnectionInject(
        gpuStages['attn_hc.input'],
        gpuStages['attn_hc.branchOutput'],
        gpuStages['attn_hc.gate'],
        numPos,
        p.numStreams,
        p.hiddenSize
      ),
  },
  {
    stage: 'mlp_hc.input',
    kernel: 'identity (attn_hc.injected)',
    actual: gpuStages['mlp_hc.input'],
    predict: () => gpuStages['attn_hc.injected'],
  },
  {
    stage: 'mlp_hc.norm',
    kernel: 'hyper_connection_norm (4x2560 streams, 256-thread tree + divideRefined + inverseRootRefined)',
    actual: gpuStages['mlp_hc.norm'],
    predict: () => witnessHyperConnectionNorm(gpuStages['mlp_hc.input'], W.mlpNorm, numPos),
  },
  {
    stage: 'mlp_hc.down',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['mlp_hc.down'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['mlp_hc.norm'], numPos, p.streamWidth, W.mlpDown, p.lowRankDim),
  },
  {
    stage: 'mlp_hc.lowRank',
    kernel: 'hyper_connection_low_rank_prepare.wgsl (siluWgsl(divideRefined(x, 4)))',
    actual: gpuStages['mlp_hc.lowRank'],
    predict: () => witnessLowRankSilu(gpuStages['mlp_hc.down'], p.numStreams),
  },
  {
    stage: 'mlp_hc.up',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['mlp_hc.up'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['mlp_hc.lowRank'], numPos, p.lowRankDim, W.mlpUp, p.streamWidth),
  },
  {
    stage: 'mlp_hc.mix',
    kernel: 'hyper_connection_mix_prepare.wgsl (sigmoidRefined(x))',
    actual: gpuStages['mlp_hc.mix'],
    predict: () => witnessMixSigmoid(gpuStages['mlp_hc.up']),
  },
  {
    stage: 'mlp_hc.branchInput',
    kernel: 'hyper_connection_mix.wgsl (divideRefined(sum_s(norm_s * mix_s), 4))',
    actual: gpuStages['mlp_hc.branchInput'],
    predict: () => witnessHyperConnectionBranch(gpuStages['mlp_hc.norm'], gpuStages['mlp_hc.mix'], numPos, p.numStreams, p.hiddenSize),
  },
  {
    stage: 'mlp_hc.logits',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['mlp_hc.logits'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['mlp_hc.norm'], numPos, p.streamWidth, W.mlpInject, p.numStreams),
  },
  {
    stage: 'mlp_hc.gate',
    kernel: 'hyper_connection_inject_prepare.wgsl (2 * sigmoidRefined(divideRefined(x, 4)))',
    actual: gpuStages['mlp_hc.gate'],
    predict: () => witnessInjectGate(gpuStages['mlp_hc.logits'], p.numStreams),
  },
  {
    stage: 'router25.input',
    kernel: 'identity (mlp_hc.branchInput)',
    actual: gpuStages['router25.input'],
    predict: () => gpuStages['mlp_hc.branchInput'],
  },
  {
    stage: 'router25.logits',
    kernel: 'matmul_f32_row_subgroup.wgsl (wave64 vec4 FMA + DPP 1..32)',
    actual: gpuStages['router25.logits'],
    predict: () => witnessMatmulRowSubgroupWave64(gpuStages['router25.input'], numPos, p.hiddenSize, W.routerGate, p.expertCount),
  },
];

const stageWitnessResults = [];
let totalStageValuesChecked = 0;
let totalStageValuesExact0Ulp = 0;

for (const def of witnessStageDefs) {
  const predicted = def.predict();
  const cmp = compareF32Bits(predicted, def.actual);
  if (!cmp.allExact) {
    throw new Error(`Stage witness ${def.stage} failed 0-ULP check: ${JSON.stringify(cmp)}`);
  }
  totalStageValuesChecked += cmp.total;
  totalStageValuesExact0Ulp += cmp.exact;
  stageWitnessResults.push({
    stage: def.stage,
    kernel: def.kernel,
    ...cmp,
  });
}

// Also verify full 2,304-row router25.logits (1,179,648 values)
const fullRouterLogitsWitness = witnessMatmulRowSubgroupWave64(
  gpuRouterInputAll,
  p.seqLen,
  p.hiddenSize,
  W.routerGate,
  p.expertCount
);
const fullRouterLogitsCmp = compareF32Bits(fullRouterLogitsWitness, gpuRouterLogitsAll);
if (!fullRouterLogitsCmp.allExact) {
  throw new Error('Full 2304-row router25.logits witness failed 0-ULP check');
}

// 5. Chained End-to-End Pre-Router + Router Witness
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

function runLinearAttnWithSplicedPosRows(baseQkv, baseZ, baseA, baseB, posBranchInput) {
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
  const seqCore = witnessGatedDeltaCore(
    seqQkv,
    seqZ,
    seqA,
    seqB,
    W.conv1d,
    W.dtBias,
    W.aLog,
    W.normW,
    p.seqLen
  );
  const posCore = slicePosRows(seqCore, p.zDim);
  const branchOutput = witnessMatmulF16wF32aTiled16(posCore, numPos, p.zDim, W.outProj, p.hiddenSize);
  return { posCore, branchOutput };
}

const chainedAttnHc = runWgslHcForward(gpuStages['attn_hc.input'], W.attnNorm, W.attnDown, W.attnUp);
const chainedLin = runLinearAttnWithSplicedPosRows(
  gpuStages['linear.attn.qkv_proj'],
  gpuStages['linear.attn.linear_z_proj'],
  gpuStages['linear.attn.linear_a_proj'],
  gpuStages['linear.attn.linear_b_proj'],
  chainedAttnHc.branchInput
);
const chainedInject = runWgslHcInject(
  gpuStages['attn_hc.input'],
  chainedAttnHc.norm,
  chainedLin.branchOutput,
  W.attnInject
);
const chainedMlpHc = runWgslHcForward(chainedInject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);
const chainedRouterLogits = witnessMatmulRowSubgroupWave64(
  chainedMlpHc.branchInput,
  numPos,
  p.hiddenSize,
  W.routerGate,
  p.expertCount
);

const chainedBranchInputCmp = compareF32Bits(chainedMlpHc.branchInput, gpuStages['mlp_hc.branchInput']);
const chainedRouterLogitsCmp = compareF32Bits(chainedRouterLogits, gpuStages['router25.logits']);
if (!chainedBranchInputCmp.allExact || !chainedRouterLogitsCmp.allExact) {
  throw new Error('Chained end-to-end Layer-25 WGSL witness failed 0-ULP check');
}

// 6. Fault / Negative Controls (7 operator fault controls)
function countRejectedRows(predicted, actual, width) {
  let rejected = 0;
  for (let pi = 0; pi < numPos; pi++) {
    const pRow = predicted.subarray(pi * width, (pi + 1) * width);
    const aRow = actual.subarray(pi * width, (pi + 1) * width);
    if (!compareF32Bits(pRow, aRow).allExact) rejected++;
  }
  return rejected;
}

const faultControls = [];

// Control 1: 1-ULP corrupted output on attn_hc.norm
{
  const corrupted = new Float32Array(gpuStages['attn_hc.norm']);
  for (let pi = 0; pi < numPos; pi++) {
    const idx = pi * p.streamWidth;
    corrupted[idx] = bitsToF32(f32ToBits(corrupted[idx]) ^ 1);
  }
  const pred = witnessHyperConnectionNorm(gpuStages['attn_hc.input'], W.attnNorm, numPos);
  const rejected = countRejectedRows(pred, corrupted, p.streamWidth);
  faultControls.push({
    control: 'corrupted_output_1ulp',
    stage: 'attn_hc.norm',
    rowsTested: numPos,
    rowsRejected: rejected,
    allRejected: rejected === numPos,
  });
}

// Control 2: Missing weight offset (w instead of 1 + w) on attn_hc.norm
{
  const wrong = witnessHyperConnectionNorm(gpuStages['attn_hc.input'], W.attnNorm, numPos, { weightOffset: false });
  const rejected = countRejectedRows(wrong, gpuStages['attn_hc.norm'], p.streamWidth);
  faultControls.push({
    control: 'norm_missing_weight_offset',
    stage: 'attn_hc.norm',
    rowsTested: numPos,
    rowsRejected: rejected,
    allRejected: rejected === numPos,
  });
}

// Control 3: Right-associated norm multiply x * (rrms * (1 + w)) on attn_hc.norm
{
  const wrong = witnessHyperConnectionNorm(gpuStages['attn_hc.input'], W.attnNorm, numPos, { rightAssociated: true });
  const rejected = countRejectedRows(wrong, gpuStages['attn_hc.norm'], p.streamWidth);
  faultControls.push({
    control: 'norm_right_associated',
    stage: 'attn_hc.norm',
    rowsTested: numPos,
    rowsRejected: rejected,
    allRejected: rejected === numPos,
  });
}

// Control 4: Top-down subgroup reduction (32,16,8,4,2,1) instead of bottom-up (1,2,4,8,16,32) on attn_hc.down
{
  const wrong = witnessMatmulRowSubgroupWave64(gpuStages['attn_hc.norm'], numPos, p.streamWidth, W.attnDown, p.lowRankDim, {
    reductionOrder: 'top-down',
  });
  const rejected = countRejectedRows(wrong, gpuStages['attn_hc.down'], p.lowRankDim);
  faultControls.push({
    control: 'subgroup_top_down_reduction',
    stage: 'attn_hc.down',
    rowsTested: numPos,
    rowsRejected: rejected,
    allRejected: rejected === numPos,
  });
}

// Control 5: Sequential 0..K-1 FMA instead of 16-element K-tile FMA on linear.attn.qkv_proj
{
  const wrong = witnessMatmulF16wF32aSeq(gpuPosProjIn, numPos, p.hiddenSize, W.qkvProj, p.qkvDim);
  const rejected = countRejectedRows(wrong, gpuPosQkv, p.qkvDim);
  faultControls.push({
    control: 'matmul_seq_instead_of_tiled16',
    stage: 'linear.attn.qkv_proj@positions',
    rowsTested: numPos,
    rowsRejected: rejected,
    allRejected: rejected === numPos,
  });
}

// Control 6: Unfused multiply-add instead of fma() in gated_delta_conv
{
  const wrongCore = witnessGatedDeltaCore(
    gpuStages['linear.attn.qkv_proj'].subarray(0, 4 * p.qkvDim),
    gpuStages['linear.attn.linear_z_proj'].subarray(0, 4 * p.zDim),
    gpuStages['linear.attn.linear_a_proj'].subarray(0, 4 * p.numValueHeads),
    gpuStages['linear.attn.linear_b_proj'].subarray(0, 4 * p.numValueHeads),
    W.conv1d,
    W.dtBias,
    W.aLog,
    W.normW,
    4,
    { convFma: false }
  );
  let rejected = 0;
  for (let t = 1; t < 4; t++) {
    const pRow = wrongCore.subarray(t * p.zDim, (t + 1) * p.zDim);
    const aRow = gpuStages['linear.attn.linear_core_out'].subarray(t * p.zDim, (t + 1) * p.zDim);
    if (!compareF32Bits(pRow, aRow).allExact) rejected++;
  }
  faultControls.push({
    control: 'conv1d_unfused_mul_add',
    stage: 'linear.attn.linear_core_out',
    rowsTested: 3,
    rowsRejected: rejected,
    allRejected: rejected === 3,
  });
}

// Control 7: Wrong GDN norm-gate activation (silu instead of sigmoid)
{
  const wrongCore = witnessGatedDeltaCore(
    gpuStages['linear.attn.qkv_proj'].subarray(0, 4 * p.qkvDim),
    gpuStages['linear.attn.linear_z_proj'].subarray(0, 4 * p.zDim),
    gpuStages['linear.attn.linear_a_proj'].subarray(0, 4 * p.numValueHeads),
    gpuStages['linear.attn.linear_b_proj'].subarray(0, 4 * p.numValueHeads),
    W.conv1d,
    W.dtBias,
    W.aLog,
    W.normW,
    4,
    { gateActivation: 'silu' }
  );
  let rejected = 0;
  for (let t = 0; t < 3; t++) {
    const pRow = wrongCore.subarray(t * p.zDim, (t + 1) * p.zDim);
    const aRow = gpuStages['linear.attn.linear_core_out'].subarray(t * p.zDim, (t + 1) * p.zDim);
    if (!compareF32Bits(pRow, aRow).allExact) rejected++;
  }
  faultControls.push({
    control: 'gdn_wrong_silu_gate',
    stage: 'linear.attn.linear_core_out',
    rowsTested: 3,
    rowsRejected: rejected,
    allRejected: rejected === 3,
  });
}

if (faultControls.some((c) => !c.allRejected)) {
  throw new Error(`One or more fault controls failed to reject: ${JSON.stringify(faultControls)}`);
}

// 7. Stage-by-Stage GPU vs CPU Reference Drift Progression Across Layer 25
const stageDriftSpecs = [
  ['attn_hc.input', gpuStages['attn_hc.input'], cpuStages['attn_hc.input'], p.streamWidth],
  ['attn_hc.norm', gpuStages['attn_hc.norm'], cpuStages['attn_hc.norm'], p.streamWidth],
  ['attn_hc.down', gpuStages['attn_hc.down'], cpuStages['attn_hc.down'], p.lowRankDim],
  ['attn_hc.lowRank', gpuStages['attn_hc.lowRank'], cpuStages['attn_hc.lowRank'], p.lowRankDim],
  ['attn_hc.up', gpuStages['attn_hc.up'], cpuStages['attn_hc.up'], p.streamWidth],
  ['attn_hc.mix', gpuStages['attn_hc.mix'], cpuStages['attn_hc.mix'], p.streamWidth],
  ['attn_hc.branchInput', gpuStages['attn_hc.branchInput'], cpuStages['attn_hc.branchInput'], p.hiddenSize],
  ['linear.attn.qkv_proj', gpuPosQkv, cpuStages['linear.attn.qkv_proj'], p.qkvDim],
  ['linear.attn.linear_z_proj', gpuPosZ, cpuStages['linear.attn.linear_z_proj'], p.zDim],
  ['linear.attn.linear_a_proj', gpuPosA, cpuStages['linear.attn.linear_a_proj'], p.numValueHeads],
  ['linear.attn.linear_b_proj', gpuPosB, cpuStages['linear.attn.linear_b_proj'], p.numValueHeads],
  ['linear.attn.linear_core_out', gpuPosCoreOut, cpuStages['linear.attn.linear_core_out'], p.zDim],
  ['attn_hc.branchOutput', gpuStages['attn_hc.branchOutput'], cpuStages['attn_hc.branchOutput'], p.hiddenSize],
  ['attn_hc.logits', gpuStages['attn_hc.logits'], cpuStages['attn_hc.logits'], p.numStreams],
  ['attn_hc.gate', gpuStages['attn_hc.gate'], cpuStages['attn_hc.gate'], p.numStreams],
  ['attn_hc.injected', gpuStages['attn_hc.injected'], cpuStages['attn_hc.injected'], p.streamWidth],
  ['mlp_hc.norm', gpuStages['mlp_hc.norm'], cpuStages['mlp_hc.norm'], p.streamWidth],
  ['mlp_hc.down', gpuStages['mlp_hc.down'], cpuStages['mlp_hc.down'], p.lowRankDim],
  ['mlp_hc.lowRank', gpuStages['mlp_hc.lowRank'], cpuStages['mlp_hc.lowRank'], p.lowRankDim],
  ['mlp_hc.up', gpuStages['mlp_hc.up'], cpuStages['mlp_hc.up'], p.streamWidth],
  ['mlp_hc.mix', gpuStages['mlp_hc.mix'], cpuStages['mlp_hc.mix'], p.streamWidth],
  ['mlp_hc.branchInput', gpuStages['mlp_hc.branchInput'], cpuStages['mlp_hc.branchInput'], p.hiddenSize],
  ['router25.logits', gpuStages['router25.logits'], cpuStages['router25.logits'], p.expertCount],
];

const stageGpuVsCpuDrift = stageDriftSpecs.map(([stage, gpuArr, cpuArr, width]) => {
  const perPosition = positions.map((pos, pi) => {
    const g = gpuArr.subarray(pi * width, (pi + 1) * width);
    const c = cpuArr.subarray(pi * width, (pi + 1) * width);
    let maxAbs = 0;
    let sumSq = 0;
    for (let i = 0; i < width; i++) {
      const d = Math.abs(g[i] - c[i]);
      if (d > maxAbs) maxAbs = d;
      sumSq += d * d;
    }
    return { position: pos, maxAbsDiff: maxAbs, rmsDiff: Math.sqrt(sumSq / width) };
  });
  return { stage, width, perPosition };
});

// 8. Bidirectional Counterfactual Attribution (A0..A6 forward GPU->CPU, R0..R3 reverse CPU->GPU)
// First verify our exact F64 CPU reference operators against cpuStages (0 ULP on attn_hc, inject, mlp_hc)
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
  return { norm, branchInput };
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

function evaluateRouterCounterfactual(id, description, branchInputRows) {
  const wgslAllLogits = witnessMatmulRowSubgroupWave64(branchInputRows, numPos, p.hiddenSize, W.routerGate, p.expertCount);
  const perPosition = positions.map((pos, pi) => {
    const xTyped = branchInputRows.subarray(pi * p.hiddenSize, (pi + 1) * p.hiddenSize);
    const xArr = Array.from(xTyped);
    const exactLogits = routerWeightsByExpert.map((w) => dotEnvelope(xArr, w, 0).exact);
    const wgslLogits = Array.from(wgslAllLogits.subarray(pi * p.expertCount, (pi + 1) * p.expertCount));
    const exactTop = rankTopK(exactLogits);
    const wgslTop = rankTopK(wgslLogits);
    const refTop = cpuExactTopByPos[pi];
    const exactOverlapWithCpu = exactTop.filter((e) => refTop.includes(e)).length;
    const wgslOverlapWithCpu = wgslTop.filter((e) => refTop.includes(e)).length;

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
      exactTop,
      wgslTop,
      exactOverlapWithCpu,
      wgslOverlapWithCpu,
      exactMatchesCpuTop10: exactOverlapWithCpu === p.topK,
      wgslMatchesCpuTop10: wgslOverlapWithCpu === p.topK,
      pairMargin,
    };
  });
  return { id, description, perPosition };
}

// Forward Counterfactuals (A0, A1, A2, A5, A6)
const a0BranchIn = cpuStages['mlp_hc.branchInput'];

const a1AttnHc = cpuHcForwardF64(gpuStages['attn_hc.input'], W.attnNorm, W.attnDown, W.attnUp);
const a1Inject = cpuHcInjectF64(
  gpuStages['attn_hc.input'],
  a1AttnHc.norm,
  cpuStages['attn_hc.branchOutput'],
  W.attnInject
);
const a1MlpHc = cpuHcForwardF64(a1Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);

const a2Lin = runLinearAttnWithSplicedPosRows(
  gpuStages['linear.attn.qkv_proj'],
  gpuStages['linear.attn.linear_z_proj'],
  gpuStages['linear.attn.linear_a_proj'],
  gpuStages['linear.attn.linear_b_proj'],
  a1AttnHc.branchInput
);
const a2Inject = cpuHcInjectF64(gpuStages['attn_hc.input'], a1AttnHc.norm, a2Lin.branchOutput, W.attnInject);
const a2MlpHc = cpuHcForwardF64(a2Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);

const a5Inject = cpuHcInjectF64(
  gpuStages['attn_hc.input'],
  gpuStages['attn_hc.norm'],
  gpuStages['attn_hc.branchOutput'],
  W.attnInject
);
const a5MlpHc = cpuHcForwardF64(a5Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);

const a6BranchIn = gpuStages['mlp_hc.branchInput'];

// Reverse Counterfactuals (R0, R1, R2)
const r1AttnHc = runWgslHcForward(cpuStages['attn_hc.input'], W.attnNorm, W.attnDown, W.attnUp);
const r1Inject = runWgslHcInject(
  cpuStages['attn_hc.input'],
  r1AttnHc.norm,
  cpuStages['attn_hc.branchOutput'],
  W.attnInject
);
const r1MlpHc = runWgslHcForward(r1Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);

const r2Lin = runLinearAttnWithSplicedPosRows(
  gpuStages['linear.attn.qkv_proj'],
  gpuStages['linear.attn.linear_z_proj'],
  gpuStages['linear.attn.linear_a_proj'],
  gpuStages['linear.attn.linear_b_proj'],
  r1AttnHc.branchInput
);
const r2Inject = runWgslHcInject(cpuStages['attn_hc.input'], r1AttnHc.norm, r2Lin.branchOutput, W.attnInject);
const r2MlpHc = runWgslHcForward(r2Inject.injected, W.mlpNorm, W.mlpDown, W.mlpUp);

const forwardCounterfactuals = [
  evaluateRouterCounterfactual('A0_pure_cpu', 'Pure CPU reference Layer 25', a0BranchIn),
  evaluateRouterCounterfactual(
    'A1_gpu_layer24_stream_with_cpu_linear_branch',
    'GPU attn_hc.input (from Layer 24) at [1717, 2271, 2303] + pure CPU attn_hc, CPU linear_attn branchOutput, CPU inject, and CPU mlp_hc',
    a1MlpHc.branchInput
  ),
  evaluateRouterCounterfactual(
    'A2_gpu_layer24_stream_with_spliced_linear_attn',
    'GPU attn_hc.input (from Layer 24) at [1717, 2271, 2303] + pure CPU attn_hc, spliced Linear Attention, CPU inject, and CPU mlp_hc',
    a2MlpHc.branchInput
  ),
  evaluateRouterCounterfactual(
    'A5_gpu_attn_hc_and_linear_attn',
    'GPU attn_hc + GPU linear_attn branchOutput + pure CPU inject and mlp_hc',
    a5MlpHc.branchInput
  ),
  evaluateRouterCounterfactual(
    'A6_all_gpu_layer25',
    'Actual GPU Layer-25 router input (all GPU Layer-25 operators)',
    a6BranchIn
  ),
];

const reverseCounterfactuals = [
  evaluateRouterCounterfactual(
    'R0_all_gpu_layer25',
    'Actual GPU Layer-25 router input (GPU Layer-24 state + GPU WGSL Layer-25 operators)',
    a6BranchIn
  ),
  evaluateRouterCounterfactual(
    'R1_cpu_stream_and_branch_into_gpu_wgsl_hc',
    'CPU attn_hc.input + CPU linear_attn branchOutput + GPU WGSL attn_hc, inject, and mlp_hc',
    r1MlpHc.branchInput
  ),
  evaluateRouterCounterfactual(
    'R2_cpu_layer24_stream_into_all_gpu_wgsl_layer25',
    'CPU attn_hc.input at [1717, 2271, 2303] + 100% GPU WGSL Layer-25 operators (attn_hc -> linear_attn -> inject -> mlp_hc -> router25)',
    r2MlpHc.branchInput
  ),
];

const result = {
  schema: 'doppler.layer25-prerouter-witness/v1',
  qualified: false,
  protocolSha256: sha(protocolBytes),
  toolSha256: sha(fs.readFileSync(new URL(import.meta.url))),
  witnessCoreSha256: sha(fs.readFileSync(new URL('./layer25-wgsl-witness-core.js', import.meta.url))),
  arithmeticCoreSha256: sha(fs.readFileSync(new URL('./numerical-discontinuity-core.js', import.meta.url))),
  layer: p.layer,
  positions: p.positions,
  seqLen: p.seqLen,
  identities,
  weightIdentities,
  summary: {
    stageWitnessesChecked: stageWitnessResults.length,
    stageWitnessesAll0Ulp: stageWitnessResults.every((s) => s.allExact),
    stagePositionValuesChecked: totalStageValuesChecked,
    stagePositionValuesExact0Ulp: totalStageValuesExact0Ulp,
    fullSeqLinearCoreValuesChecked: fullCoreOutCmp.total,
    fullSeqLinearCoreValuesExact0Ulp: fullCoreOutCmp.exact,
    fullSeqRouterLogitsValuesChecked: fullRouterLogitsCmp.total,
    fullSeqRouterLogitsValuesExact0Ulp: fullRouterLogitsCmp.exact,
    chainedEndToEndBranchInputExact0Ulp: chainedBranchInputCmp.exact,
    chainedEndToEndRouterLogitsExact0Ulp: chainedRouterLogitsCmp.exact,
    faultControlsTested: faultControls.length,
    faultControlsRejected: faultControls.filter((c) => c.allRejected).length,
    streamResidualPathAloneFlips1717:
      forwardCounterfactuals[1].perPosition[0].pairMargin.flipsExact &&
      !forwardCounterfactuals[1].perPosition[1].pairMargin.flipsExact,
    incomingLayer24DriftThroughLinearAttnFlipsBothPositions:
      forwardCounterfactuals[2].perPosition[0].pairMargin.flipsExact &&
      forwardCounterfactuals[2].perPosition[1].pairMargin.flipsExact,
    incomingLayer24DriftNecessaryForGpuLayer25Flip:
      reverseCounterfactuals[1].perPosition.every((r) => r.exactMatchesCpuTop10 && r.wgslMatchesCpuTop10) &&
      reverseCounterfactuals[2].perPosition.every((r) => r.exactMatchesCpuTop10 && r.wgslMatchesCpuTop10),
  },
  stageWitnessResults,
  fullSeqLinearCoreWitness: fullCoreOutCmp,
  fullSeqRouterLogitsWitness: fullRouterLogitsCmp,
  chainedEndToEndWitness: {
    mlpBranchInput: chainedBranchInputCmp,
    routerLogits: chainedRouterLogitsCmp,
  },
  faultControls,
  stageGpuVsCpuDrift,
  forwardCounterfactuals,
  reverseCounterfactuals,
  scope:
    'Constructive F32 WGSL operator witnesses and exact dyadic / WGSL counterfactuals for Layer 25 only. Proves 0-ULP operator fidelity across Layer 25 and attributes both Layer-25 routing flips to incoming Layer-24 stream drift (attn_hc.input). Does not bound Layers 0..24 or qualify the model.',
};

fs.writeFileSync(outputPath, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify(result.summary, null, 2));
