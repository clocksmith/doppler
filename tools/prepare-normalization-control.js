#!/usr/bin/env node
// Reproduce the frozen synthetic fixture, or package a physical replay for the
// same offline witness analyzer. Never invokes a GPU or changes model inputs.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { gzipSync } from 'node:zlib';
import { halfValue, nearestHalf } from './numerical-discontinuity-core.js';

const sha = (bytes) => crypto.createHash('sha256').update(bytes).digest('hex');
const read = (file) => JSON.parse(fs.readFileSync(file));
const [mode, ...args] = process.argv.slice(2);
if (mode === 'fixture') {
  const [evidencePath, protocolPath, output] = args;
  if (!output) throw new Error('Usage: prepare-normalization-control.js fixture evidence.json heldout-protocol.json output.json');
  const e = read(evidencePath), p = read(protocolPath);
  const weights = Buffer.from(e.weightsF16, 'base64');
  if (sha(weights) !== e.weightsSha256 || p.width !== 1024 || p.rows !== 32 || p.seed !== '0x6e7149ab') throw new Error('Frozen fixture identity mismatch');
  let seed = Number(p.seed);
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  const input = Buffer.alloc(p.rows * p.width * 4), expected = Buffer.alloc(p.rows * p.width * 8);
  const observed = Buffer.alloc(input.length);
  for (let row = 0; row < p.rows; row++) {
    const values = Array.from({ length: p.width }, () => halfValue(nearestHalf((random() - 0.5) * 2 ** ((row % 16) - 8))));
    const sum = values.reduce((total, x) => total + x * x, 0);
    const inverse = 1 / Math.sqrt(sum / p.width + 1e-6);
    for (let i = 0; i < p.width; i++) {
      const offset = row * p.width + i;
      const y = values[i] * inverse * (1 + halfValue(weights.readUInt16LE(i * 2)));
      input.writeFloatLE(values[i], offset * 4); expected.writeDoubleLE(y, offset * 8); observed.writeFloatLE(y, offset * 4);
    }
  }
  const fixture = { source: p.source, rows: p.rows, width: p.width, epsilon: 1e-6,
    pipeline: { entry: 'main', constants: { RMS_NORM_OFFSET: 1, PRE_RESIDUAL: 0, OUTPUT_PRENORM: 0, WEIGHT_IS_F16: 1 } },
    input: input.toString('base64'), weight: weights.toString('base64'), expected: expected.toString('base64'), observed: observed.toString('base64') };
  const bytes = Buffer.from(JSON.stringify(fixture));
  if (sha(bytes) !== p.fixtureSha256) throw new Error('Regenerated fixture differs from frozen bytes');
  fs.writeFileSync(output, bytes);
  console.log(JSON.stringify({ fixtureSha256: sha(bytes), bytes: bytes.length }));
} else if (mode === 'bundle') {
  const [protocolPath, fixturePath, replayPath, originalProtocolPath, bundlePath, analysisPath] = args;
  if (!analysisPath) throw new Error('Usage: prepare-normalization-control.js bundle heldout-protocol fixture replay original-protocol output.gz analysis.json');
  const p = read(protocolPath), f = read(fixturePath), g = read(replayPath), original = read(originalProtocolPath);
  if (sha(fs.readFileSync(fixturePath)) !== p.fixtureSha256 || g.fixtureSha256 !== p.fixtureSha256 ||
      g.shaderSha256 !== p.shaderSha256 || !g.executed || g.errors.length || g.values.length !== p.rows * p.width ||
      sha(Buffer.from(Float32Array.from(g.values).buffer)) !== g.valuesSha256) throw new Error('Replay identity mismatch');
  const x = Buffer.from(f.input, 'base64'), ref = Buffer.from(f.expected, 'base64');
  const values = Array.from({ length: f.rows * f.width }, (_, i) => x.readFloatLE(i * 4));
  const bundle = { first: [{ op: 'embed.out', values, referenceValues: values },
    { op: 'layer.0.attn.post_input_norm', values: g.values, referenceValues: Array.from({ length: values.length }, (_, i) => ref.readDoubleLE(i * 8)) }] };
  const bytes = gzipSync(JSON.stringify(bundle));
  original.inputBundleSha256 = sha(bytes); original.shape = [f.rows, f.width];
  original.hypothesis = 'Apply the already frozen arithmetic models to new synthetic operator rows on physical Metal';
  original.parentProtocolSha256 = sha(fs.readFileSync(originalProtocolPath));
  original.predeclaredHeldoutProtocolSha256 = sha(fs.readFileSync(protocolPath));
  original.limitations = ['Synthetic operator inputs, not held-out prompts', 'Physical Metal; no AMD operator rerun', 'No complete model qualification'];
  fs.writeFileSync(bundlePath, bytes); fs.writeFileSync(analysisPath, JSON.stringify(original, null, 2) + '\n');
  console.log(JSON.stringify({ bundleSha256: sha(bytes), analysisSha256: sha(fs.readFileSync(analysisPath)) }));
} else throw new Error('Expected fixture or bundle mode');
