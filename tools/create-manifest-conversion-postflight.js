#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createManifestConversionPostflightReceipt } from '../src/converter/manifest-conversion-postflight.js';
import { parseManifest } from '../src/formats/rdrr/parsing.js';

function parseArgs(argv) {
  const options = { check: false };
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--policy') options.policy = argv[++index];
    else if (token === '--check') options.check = true;
    else throw new Error(`Unknown argument "${token}".`);
  }
  if (!options.policy) throw new Error('Usage: --policy <path> [--check]');
  return options;
}

function workspacePath(repoRoot, value, label) {
  if (typeof value !== 'string' || !value.trim() || path.isAbsolute(value)) {
    throw new Error(`${label} must be a workspace-relative path.`);
  }
  const resolved = path.resolve(repoRoot, value);
  const relative = path.relative(repoRoot, resolved);
  if (relative === '..' || relative.startsWith(`..${path.sep}`)) {
    throw new Error(`${label} must remain inside the workspace.`);
  }
  return resolved;
}

async function readJsonWithDigest(filePath) {
  const bytes = await fs.readFile(filePath);
  return {
    value: JSON.parse(bytes.toString('utf8')),
    digest: `sha256:${createHash('sha256').update(bytes).digest('hex')}`,
  };
}

async function observeFile(filePath) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(filePath)) hash.update(chunk);
  const stats = await fs.stat(filePath);
  return { size: stats.size, digest: `sha256:${hash.digest('hex')}` };
}

async function mapConcurrent(values, concurrency, transform) {
  const results = new Array(values.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, async () => {
    while (cursor < values.length) {
      const index = cursor++;
      results[index] = await transform(values[index], index);
    }
  }));
  return results;
}

const options = parseArgs(process.argv.slice(2));
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const policyPath = workspacePath(repoRoot, options.policy, 'policy');
const policy = JSON.parse(await fs.readFile(policyPath, 'utf8'));
const modelDir = workspacePath(repoRoot, policy.modelDir, 'policy.modelDir');
const [conversionConfigFile, conversionReportFile, manifestFile, preflightFile] = await Promise.all([
  readJsonWithDigest(workspacePath(repoRoot, policy.conversionConfig, 'policy.conversionConfig')),
  readJsonWithDigest(workspacePath(repoRoot, policy.conversionReport, 'policy.conversionReport')),
  readJsonWithDigest(workspacePath(repoRoot, policy.manifest, 'policy.manifest')),
  readJsonWithDigest(workspacePath(repoRoot, policy.preflightReceipt, 'policy.preflightReceipt')),
]);
const manifest = parseManifest(JSON.stringify(manifestFile.value));
const concurrency = Number(policy.hashConcurrency);
if (!Number.isInteger(concurrency) || concurrency < 1) {
  throw new Error('policy.hashConcurrency must be a positive integer.');
}
const shardObservations = await mapConcurrent(manifest.shards, concurrency, async (shard) => ({
  index: shard.index,
  filename: shard.filename,
  ...await observeFile(workspacePath(modelDir, shard.filename, `manifest.shards[${shard.index}].filename`)),
}));
if (!Array.isArray(policy.artifacts) || policy.artifacts.length < 1) {
  throw new Error('policy.artifacts must be a non-empty array.');
}
const artifactObservations = await mapConcurrent(policy.artifacts, concurrency, async (artifact, index) => {
  if (!artifact || typeof artifact !== 'object' || Array.isArray(artifact)) {
    throw new Error(`policy.artifacts[${index}] must be an object.`);
  }
  const artifactPath = workspacePath(modelDir, artifact.path, `policy.artifacts[${index}].path`);
  return { role: artifact.role, path: artifact.path, ...await observeFile(artifactPath) };
});
const receipt = createManifestConversionPostflightReceipt({
  conversionConfig: conversionConfigFile.value,
  conversionReport: conversionReportFile.value,
  conversionReportDigest: conversionReportFile.digest,
  manifest,
  manifestDigest: manifestFile.digest,
  preflightReceipt: preflightFile.value,
  shardObservations,
  artifactObservations,
  policy,
});
const outputPath = workspacePath(repoRoot, policy.output, 'policy.output');
const outputText = `${JSON.stringify(receipt, null, 2)}\n`;
if (options.check) {
  const observed = await fs.readFile(outputPath, 'utf8');
  if (observed !== outputText) throw new Error(`Manifest conversion postflight drifted: ${policy.output}.`);
} else {
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  await fs.writeFile(outputPath, outputText);
}
console.log(`${receipt.modelId}/${receipt.entryPointId}: ${receipt.physicalClosure.shardCount} shards postflighted`);
