#!/usr/bin/env node
// Frozen F32-KV router inputs only. Does not open a GPU or change qualification.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readNpyF32 } from './compare-source-boundaries.js';
import { halfValue, dotEnvelope } from './numerical-discontinuity-core.js';

const [protocolPath, artifact, qualify, sourceRoot, output] = process.argv.slice(2);
if (!output) throw new Error('Usage: analyze-router-input-drift.js protocol artifact qualify source-trace output.json');
const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const protocolBytes = fs.readFileSync(protocolPath), p = JSON.parse(protocolBytes);
const identities = [];
function checked(file, expected) {
  const bytes = fs.readFileSync(file);
  if (!expected || sha(bytes) !== expected) throw new Error(`Identity mismatch: ${path.basename(file)}`);
  identities.push({ file: path.basename(file), sha256: expected, bytes: bytes.length });
  return bytes;
}
function range(file, offset, size) {
  const fd = fs.openSync(file, 'r');
  try {
    const bytes = Buffer.alloc(size);
    let count = 0;
    while (count < size) {
      const n = fs.readSync(fd, bytes, count, size - count, offset + count);
      if (!n) throw new Error('Truncated weight span');
      count += n;
    }
    return bytes;
  } finally { fs.closeSync(fd); }
}
const manifest = JSON.parse(checked(path.join(artifact, 'manifest.json'), p.manifestSha256));
const observation = JSON.parse(checked(path.join(qualify, p.observation), p.observationSha256));
const comparison = JSON.parse(checked(path.join(qualify, p.comparison), p.comparisonSha256));
const source = JSON.parse(checked(path.join(sourceRoot, 'receipt.json'), p.sourceReceiptSha256));
if (!observation.completed || !source.completed || observation.manifestSha256 !== p.manifestSha256 ||
    source.manifestSha256 !== p.manifestSha256 || comparison.gpuSha256 !== p.observationSha256 ||
    comparison.sourceTraceSha256 !== p.sourceReceiptSha256 || comparison.firstChangedLayer !== p.layer) {
  throw new Error('Unmatched or incomplete source chain');
}
const record = observation.rows[0].routerObservations.find((r) => r.layer === p.layer);
const input = checked(path.join(qualify, path.basename(record.input.file)), record.input.sha256);
const logits = checked(path.join(qualify, path.basename(record.logits.file)), record.logits.sha256);
const filename = `router-${p.layer}.npz`, fixture = path.join(sourceRoot, filename);
checked(fixture, source.files[filename]);
const npy = (name) => readNpyF32(execFileSync('unzip', ['-p', fixture, `${name}.npy`], { maxBuffer: 40 * 1024 * 1024 })).values;
const sourceInput = npy('input'), sourceLogits = npy('logits');
const descriptor = manifest.tensors[`model.language_model.layers.${p.layer}.mlp.gate.weight`];
if (descriptor.dtype !== 'F16' || descriptor.shape.join(',') !== `${p.expertCount},${p.width}`) throw new Error('Router geometry mismatch');
const spans = descriptor.spans ?? [{ shardIndex: descriptor.shard, offset: descriptor.offset, size: descriptor.size }];
const raw = Buffer.concat(spans.map((s) => range(path.join(artifact, manifest.shards[s.shardIndex].filename), s.offset, s.size)));
if (sha(raw) !== p.weightsSha256 || raw.length !== p.expertCount * p.width * 2) throw new Error('Router weight identity mismatch');
if (sourceInput.length * 4 !== input.length || sourceLogits.length * 4 !== logits.length ||
    input.length / p.width !== logits.length / p.expertCount) throw new Error('Capture shape mismatch');
const weights = Array.from({ length: p.expertCount }, (_, expert) =>
  Array.from({ length: p.width }, (_, i) => halfValue(raw.readUInt16LE((expert * p.width + i) * 2))));
const rank = (values) => values.map((value, expert) => ({ value, expert })).sort((a, b) => b.value - a.value || a.expert - b.expert).slice(0, p.topK).map((r) => r.expert);
const difference = (a, b) => a.filter((x) => !b.includes(x));
const rows = [];
for (const position of p.positions) {
  if (!Number.isInteger(position) || position < 0 || (position + 1) * p.width > sourceInput.length) throw new Error('Invalid position');
  const x = Array.from({ length: p.width }, (_, i) => input.readFloatLE((position * p.width + i) * 4));
  const ref = Array.from(sourceInput.subarray(position * p.width, (position + 1) * p.width));
  const y = Array.from({ length: p.expertCount }, (_, e) => logits.readFloatLE((position * p.expertCount + e) * 4));
  const refY = Array.from(sourceLogits.subarray(position * p.expertCount, (position + 1) * p.expertCount));
  const gpu = weights.map((w, e) => dotEnvelope(x, w, y[e]));
  const cpu = weights.map((w, e) => dotEnvelope(ref, w, refY[e]));
  const gpuTop = rank(y), sourceTop = rank(refY), exactGpuTop = rank(gpu.map((r) => r.exact)), exactSourceTop = rank(cpu.map((r) => r.exact));
  const pairs = [];
  for (const sourceExpert of difference(sourceTop, gpuTop)) for (const gpuExpert of difference(gpuTop, sourceTop)) {
    const margin = (values) => values[sourceExpert] - values[gpuExpert];
    const sourceExactMargin = margin(cpu.map((r) => r.exact)), gpuExactMargin = margin(gpu.map((r) => r.exact));
    pairs.push({ sourceExpert, gpuExpert, sourceStoredMargin: margin(refY), gpuStoredMargin: margin(y),
      sourceExactMargin, gpuExactMargin, upstreamMarginChange: gpuExactMargin - sourceExactMargin,
      sourceProjectionMarginError: margin(refY) - sourceExactMargin, gpuProjectionMarginError: margin(y) - gpuExactMargin,
      orderingReversesOnExactInputs: sourceExactMargin > 0 && gpuExactMargin < 0,
      experts: [sourceExpert, gpuExpert].map((expert) => ({ expert, sourceStored: refY[expert], gpuStored: y[expert],
        sourceExact: cpu[expert].exact, gpuExact: gpu[expert].exact })) });
  }
  rows.push({ position, inputMaxAbsolute: Math.max(...x.map((v, i) => Math.abs(v - ref[i]))),
    gpuTop, sourceTop, exactGpuTop, exactSourceTop, pairs,
    gpuProjectionMaxError: Math.max(...gpu.map((r) => r.error)), sourceProjectionMaxError: Math.max(...cpu.map((r) => r.error)),
    gpuOutsideLocalEnvelope: gpu.filter((r) => !r.within).length, sourceOutsideLocalEnvelope: cpu.filter((r) => !r.within).length,
    upstreamStatus: 'No independent upstream bound; not a qualification decision' });
}
const result = { schema: 'doppler.router-input-drift/v1', qualified: false, protocolSha256: sha(protocolBytes),
  toolSha256: sha(fs.readFileSync(new URL(import.meta.url))),
  arithmeticSha256: sha(fs.readFileSync(new URL('./numerical-discontinuity-core.js', import.meta.url))),
  identities, weightSha256: sha(raw), layer: p.layer, rows,
  scope: 'Exact dyadic sums, reported rounded to F64. Raw-logit ranking, no physical softmax/top-k replay. Local envelopes are not acceptance.' };
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ layer: p.layer, rows }));
