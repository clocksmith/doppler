#!/usr/bin/env node
// Reads historical captures only; does not open a GPU or mutate a model.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { gzipSync } from 'node:zlib';
import { readNpyF32 } from './compare-source-boundaries.js';
import { halfValue } from './numerical-discontinuity-core.js';

const [protocolPath, artifact, qualify, splitRoot, indexPath, weightAuditPath, output] = process.argv.slice(2);
if (!output) throw new Error('Usage: collect-numerical-discontinuity.js protocol artifact qualify split-root raw-index weight-audit output.gz');
const sha = (data) => crypto.createHash('sha256').update(data).digest('hex');
const protocolBytes = fs.readFileSync(protocolPath);
const protocol = JSON.parse(protocolBytes);
const identities = [];
function checked(file, expected) {
  const bytes = fs.readFileSync(file);
  const actual = sha(bytes);
  if (expected && actual !== expected) throw new Error(`Identity mismatch: ${path.basename(file)}`);
  identities.push({ file: path.basename(file), bytes: bytes.length, sha256: actual });
  return bytes;
}
function range(file, offset, length) {
  const fd = fs.openSync(file, 'r');
  try {
    const bytes = Buffer.alloc(length);
    let read = 0;
    while (read < length) {
      const n = fs.readSync(fd, bytes, read, length - read, offset + read);
      if (!n) throw new Error('Truncated capture');
      read += n;
    }
    return bytes;
  } finally { fs.closeSync(fd); }
}
function hashFile(file, expected) {
  const fd = fs.openSync(file, 'r');
  const hash = crypto.createHash('sha256');
  const buffer = Buffer.alloc(4 * 1024 * 1024);
  let length = 0;
  try {
    for (;;) {
      const n = fs.readSync(fd, buffer);
      if (!n) break;
      hash.update(buffer.subarray(0, n)); length += n;
    }
  } finally { fs.closeSync(fd); }
  const actual = hash.digest('hex');
  if (actual !== expected) throw new Error(`Capture digest mismatch: ${path.basename(file)}`);
  identities.push({ file: path.relative(splitRoot, file), bytes: length, sha256: actual });
}
const manifest = JSON.parse(checked(path.join(artifact, 'manifest.json'), protocol.flashNext.manifestSha256));
const observation = JSON.parse(checked(path.join(qualify, protocol.flashNext.observation), protocol.flashNext.observationSha256));
const sourceRoot = path.join(qualify, 'e-router-prefix-control');
const source = JSON.parse(checked(path.join(sourceRoot, 'receipt.json'), protocol.flashNext.prefixReceiptSha256));
if (source.manifestSha256 !== protocol.flashNext.manifestSha256 || source.inputsSha256 !== protocol.flashNext.inputsSha256 || observation.inputsSha256 !== source.inputsSha256) throw new Error('Model/input identity mismatch');
const audit = JSON.parse(checked(weightAuditPath, protocol.flashNext.weightAuditSha256));
const routers = [];
for (const layer of protocol.flashNext.layers) {
  const record = observation.rows[0].routerObservations.find((r) => r.layer === layer);
  const input = checked(path.join(qualify, path.basename(record.input.file)), record.input.sha256);
  const logits = checked(path.join(qualify, path.basename(record.logits.file)), record.logits.sha256);
  const fixture = path.join(sourceRoot, `router-${layer}.npz`);
  checked(fixture, source.files[path.basename(fixture)]);
  const npy = (name) => readNpyF32(execFileSync('unzip', ['-p', fixture, `${name}.npy`], { maxBuffer: 40 * 1024 * 1024 })).values;
  const sourceInput = npy('input');
  const sourceLogits = npy('logits');
  const descriptor = manifest.tensors[`model.language_model.layers.${layer}.mlp.gate.weight`];
  if (descriptor.dtype !== 'F16' || descriptor.shape.join(',') !== '512,2560') throw new Error('Unsupported router geometry');
  const spans = descriptor.spans ?? [{ shardIndex: descriptor.shard, offset: descriptor.offset, size: descriptor.size }];
  const raw = Buffer.concat(spans.map((s) => range(path.join(artifact, manifest.shards[s.shardIndex].filename), s.offset, s.size)));
  const expanded = Buffer.alloc(raw.length * 2);
  for (let i = 0; i < raw.length / 2; i++) expanded.writeFloatLE(halfValue(raw.readUInt16LE(i * 2)), i * 4);
  if (sha(expanded) !== audit.rows.find((r) => r.layer === layer).expandedSha256) throw new Error('Historical router weight mismatch');
  const rows = protocol.flashNext.positions.map((position) => ({ position,
    input: Array.from({ length: 2560 }, (_, i) => input.readFloatLE((position * 2560 + i) * 4)),
    logits: Array.from({ length: 512 }, (_, i) => logits.readFloatLE((position * 512 + i) * 4)),
    sourceInput: Array.from(sourceInput.subarray(position * 2560, (position + 1) * 2560)),
    sourceLogits: Array.from(sourceLogits.subarray(position * 512, (position + 1) * 512)),
  }));
  routers.push({ layer, weightsF16: raw.toString('base64'), weightsSha256: sha(raw), rows });
}
const rawIndexBytes = checked(indexPath, protocol.kv.rawIndexSha256);
const rawIndex = JSON.parse(rawIndexBytes);
const captureNames = ['capture-plain', 'capture-paired', 'capture-paired-probefix'];
const captures = {};
for (const name of captureNames) {
  const directory = path.join(splitRoot, 'out', name);
  const reportEntry = rawIndex.files.find((r) => r.path === `out/${name}/report.json`);
  const report = JSON.parse(checked(path.join(directory, 'report.json'), reportEntry.sha256));
  for (const file of ['prefill.f64', 'cache-after-prefill.f16']) {
    hashFile(path.join(directory, file), rawIndex.files.find((r) => r.path === `out/${name}/${file}`).sha256);
  }
  captures[name] = { directory, report };
}
function operator(name, op) {
  const { directory, report } = captures[name];
  const row = report.captures.prefill.find((r) => r.opId === op);
  if (!row || row.tokens !== 184 || row.length !== row.width * 184) throw new Error(`Incomplete capture ${name}/${op}`);
  if (name !== 'capture-plain' && row.precision !== 'two-f32') throw new Error('Partial paired probe is forbidden');
  const raw = range(path.join(directory, 'prefill.f64'), row.offset * 8, row.length * 8);
  return Array.from({ length: row.length }, (_, i) => raw.readDoubleLE(i * 8));
}
function cache(name, layer, kind) {
  const { directory, report } = captures[name];
  let offset = 0;
  for (const row of report.caches['cache-after-prefill']) {
    const count = row.seqLen * row.numHeads * row.headDim;
    if (row.layer === layer) {
      const bytes = range(path.join(directory, 'cache-after-prefill.f16'), (offset + (kind === 'value' ? count : 0)) * 2, count * 2);
      if (sha(bytes) !== row[`${kind}Sha256`]) throw new Error('Cache slice hash mismatch');
      return Array.from({ length: count }, (_, i) => bytes.readUInt16LE(i * 2));
    }
    offset += 2 * count;
  }
  throw new Error('Missing cache layer');
}
const kv = [];
for (const layer of protocol.kv.layers) for (const kind of protocol.kv.kinds) {
  const op = `layer.${layer}.attn.${kind === 'key' ? 'k_rope' : 'v_proj'}`;
  kv.push({ layer, kind, values: operator('capture-plain', op), stored: cache('capture-plain', layer, kind),
    referenceValues: operator('capture-paired-probefix', op), referenceStored: cache('capture-paired-probefix', layer, kind) });
}
const first = ['embed.out', 'layer.0.attn.post_input_norm'].map((op) => ({ op,
  values: operator('capture-plain', op), referenceValues: operator('capture-paired', op) }));
const bundle = { schema: 'doppler.discontinuity-inputs/v1', protocolSha256: sha(protocolBytes),
  observationIdentity: { runtimeRevision: observation.runtimeRevision, runtimeSourceSha256: observation.runtimeSourceSha256, runSha256: observation.runSha256, referenceSha256: observation.referenceSha256, runtimeConfig: observation.runtimeConfig },
  sourceIdentity: source, splitIdentity: captures['capture-plain'].report,
  identities, routers, kv, first };
fs.writeFileSync(output, gzipSync(JSON.stringify(bundle)));
console.log(JSON.stringify({ output: path.basename(output), sha256: sha(fs.readFileSync(output)), bytes: fs.statSync(output).size, protocolSha256: bundle.protocolSha256 }));
