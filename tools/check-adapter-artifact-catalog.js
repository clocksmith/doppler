#!/usr/bin/env node

import { createHash } from 'node:crypto';
import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

import { validateAdapterArtifactRecord } from '../src/experimental/adapters/artifact-contract.js';

const REPO_ROOT = path.resolve(import.meta.dirname, '..');
const CATALOG_PATH = path.join(REPO_ROOT, 'models/adapters/catalog.json');
const VERIFY_LOCAL_BYTES = process.argv.includes('--verify-local-bytes');
const JSON_OUTPUT = process.argv.includes('--json');

const sha256 = (bytes) => `sha256:${createHash('sha256').update(bytes).digest('hex')}`;

const readJson = async (filePath) => JSON.parse(await readFile(filePath, 'utf8'));

const verifyFileIdentity = async (relativePath, expectedSha256, expectedBytes = null) => {
  const absolutePath = path.resolve(REPO_ROOT, relativePath);
  if (!absolutePath.startsWith(`${REPO_ROOT}${path.sep}`)) {
    throw new Error(`${relativePath}: path escapes repository root`);
  }
  const bytes = await readFile(absolutePath);
  if (expectedBytes != null && bytes.byteLength !== expectedBytes) {
    throw new Error(`${relativePath}: expected ${expectedBytes} bytes, observed ${bytes.byteLength}`);
  }
  const observedSha256 = sha256(bytes);
  if (observedSha256 !== expectedSha256) {
    throw new Error(`${relativePath}: expected ${expectedSha256}, observed ${observedSha256}`);
  }
  return { absolutePath, bytes, observedSha256 };
};

const catalog = await readJson(CATALOG_PATH);
const failures = [];
const results = [];
const ids = new Set();

if (catalog.schema !== 'doppler.adapter-artifact-catalog/v1') {
  failures.push(`catalog.schema must be doppler.adapter-artifact-catalog/v1`);
}
if (!Array.isArray(catalog.artifacts)) failures.push('catalog.artifacts must be an array');

for (const entry of catalog.artifacts || []) {
  const result = { artifactId: entry.artifactId || null, valid: false, localBytesVerified: false };
  try {
    if (ids.has(entry.artifactId)) throw new Error(`duplicate artifactId ${entry.artifactId}`);
    ids.add(entry.artifactId);
    const manifestIdentity = await verifyFileIdentity(
      entry.adapterManifestPath,
      entry.adapterManifestSha256
    );
    const adapterManifest = JSON.parse(manifestIdentity.bytes.toString('utf8'));
    const {
      adapterManifestPath,
      adapterManifestSha256,
      localWeightsPath,
      ...record
    } = entry;
    const validation = validateAdapterArtifactRecord({ ...record, adapterManifest });
    if (!validation.valid) {
      throw new Error(validation.errors.map((error) => `${error.field}: ${error.message}`).join('; '));
    }
    if (VERIFY_LOCAL_BYTES) {
      await verifyFileIdentity(localWeightsPath, entry.weights.sha256, entry.weights.bytes);
      result.localBytesVerified = true;
    } else if (localWeightsPath) {
      const absoluteWeightsPath = path.resolve(REPO_ROOT, localWeightsPath);
      const weightsStat = await stat(absoluteWeightsPath);
      result.localBytesMaterialized = weightsStat.size === entry.weights.bytes;
    }
    for (const evidence of entry.evidence) {
      await verifyFileIdentity(evidence.path, evidence.sha256);
    }
    result.valid = true;
  } catch (error) {
    result.error = error.message;
    failures.push(`${entry.artifactId || '<unknown>'}: ${error.message}`);
  }
  results.push(result);
}

const report = {
  schema: catalog.schema,
  catalogPath: path.relative(REPO_ROOT, CATALOG_PATH),
  valid: failures.length === 0,
  artifactCount: results.length,
  verifyLocalBytes: VERIFY_LOCAL_BYTES,
  failures,
  artifacts: results,
};

if (JSON_OUTPUT) {
  process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
} else if (report.valid) {
  process.stdout.write(`Adapter artifact catalog valid (${report.artifactCount} artifacts).\n`);
} else {
  process.stderr.write(`Adapter artifact catalog invalid:\n${failures.map((failure) => `- ${failure}`).join('\n')}\n`);
}

if (!report.valid) process.exitCode = 1;
