// Local-file storage observation, not a model-loading or GPU benchmark.
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { createNodeFileArtifactStorageContext } from '../../src/storage/artifact-storage-context.js';
import { computeSampleStats } from '../../src/debug/stats.js';
const output = process.argv[2];
if (!output) throw new Error('Usage: node tools/experiments/shard-hash-load.js <receipt.json>');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const fixture = Buffer.alloc(64 * 1024 * 1024, 0x5a);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const temporary = await fs.mkdtemp(path.join(os.tmpdir(), 'shard-hash-observation-'));
const samples = [];
try {
  await fs.writeFile(path.join(temporary, 'weights.bin'), fixture);
  const manifest = { modelId: 'synthetic-storage-only', hashAlgorithm: 'sha256', shards: [{ filename: 'weights.bin', size: fixture.length, hash: digest(fixture) }] };
  for (let pair = -1; pair < 6; pair++) {
    for (const verifyHashes of (pair % 2 ? [true, false] : [false, true])) {
      const context = createNodeFileArtifactStorageContext(temporary, manifest, { verifyHashes });
      try {
        const start = performance.now();
        const bytes = await context.loadShard(0);
        const durationMs = performance.now() - start;
        assert(Buffer.from(bytes).equals(fixture));
        assert(Number.isFinite(durationMs) && durationMs > 0);
        samples.push({ pair, warmup: pair < 0, verifyHashes, durationMs, outputMatches: true });
      } finally { await context.close?.(); }
    }
  }
  const summary = Object.fromEntries([false, true].map(verify => [String(verify), computeSampleStats(samples.filter(s => !s.warmup && s.verifyHashes === verify).map(s => s.durationMs), { outlierPolicy: 'none' })]));
  const receipt = { schema: 'doppler.storage-hash-observation/v1', timestamp: new Date().toISOString(), sourceCommit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(), scriptSha256: digest(await fs.readFile(fileURLToPath(import.meta.url))), command: 'node tools/experiments/shard-hash-load.js <receipt.json>', environment: { platform: process.platform, arch: process.arch, cpu: os.cpus()[0]?.model, node: process.version, backend: 'node-file', gpu: null, browser: null }, workload: { bytes: fixture.length, seed: 'constant-0x5a', sha256: manifest.shards[0].hash, pairedRuns: 6, warmupPairs: 1, cache: 'OS page cache primed; no cache flush', timing: 'loadShard only; output equality checked outside timer' }, samples, summary, limitations: ['One process on one host during ordinary development activity.', 'Synthetic 64 MiB local file with SHA-256; no network, OPFS, full model or GPU measured.', 'Explicit verification opt-out is a control, not a recommended loading policy.', 'Retains all samples; no inference throughput or cross-device claim.'] };
  await fs.mkdir(path.dirname(path.resolve(output)), { recursive: true });
  await fs.writeFile(output, JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ sourceCommit: receipt.sourceCommit, uncheckedMedianMs: summary.false.median, verifiedMedianMs: summary.true.median, samples: samples.length }));
} finally { await fs.rm(temporary, { recursive: true, force: true }); }
