#!/usr/bin/env node

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import kernelRegistry from '../src/config/kernels/registry.json' with { type: 'json' };
import { KERNEL_REF_CONTENT_DIGESTS } from '../src/config/kernels/kernel-ref-digests.js';
import { auditManifestExecutionRouting } from '../src/tooling/execution-routing-audit.js';
import { computeCanonicalSha256 } from '../src/utils/canonical-hash.js';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const modelsRoot = resolve(repoRoot, 'models/local');
const outputPath = resolve(repoRoot, 'benchmarks/kernels/execution-routing-audit.json');
const checkOnly = process.argv.includes('--check');

async function findManifests(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const paths = [];
  for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) {
      paths.push(...await findManifests(path));
    } else if (entry.isFile() && entry.name === 'manifest.json') {
      paths.push(path);
    }
  }
  return paths;
}

const audits = [];
for (const manifestPath of await findManifests(modelsRoot)) {
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  audits.push({
    manifestPath: relative(repoRoot, manifestPath).replaceAll('\\', '/'),
    ...auditManifestExecutionRouting(manifest, kernelRegistry, KERNEL_REF_CONTENT_DIGESTS),
  });
}
audits.sort((left, right) => left.manifestPath.localeCompare(right.manifestPath));
const integrityFailures = audits.flatMap((audit) => audit.integrity.filter(
  (entry) => entry.status !== 'verified'
).map((entry) => ({
  manifestPath: audit.manifestPath,
  kernelId: entry.kernelId,
  status: entry.status,
})));
const core = {
  schema: 'doppler.execution-routing-audit-index/v1',
  registryDigest: computeCanonicalSha256(kernelRegistry),
  manifestCount: audits.length,
  integrityFailures,
  opportunityCount: audits.reduce((sum, audit) => sum + audit.opportunities.length, 0),
  audits,
};
const report = { ...core, digest: computeCanonicalSha256(core) };
const output = `${JSON.stringify(report, null, 2)}\n`;

if (checkOnly) {
  const current = await readFile(outputPath, 'utf8').catch(() => null);
  if (current !== output) {
    throw new Error('Execution routing audit is stale. Run: npm run routing:audit');
  }
  console.log(
    `execution routing audit: current (${audits.length} manifests, ` +
    `${report.opportunityCount} calibration opportunities, ` +
    `${integrityFailures.length} surfaced integrity failures)`
  );
} else {
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, output);
  console.log(
    `execution routing audit: wrote ${relative(repoRoot, outputPath)} ` +
    `(${audits.length} manifests, ${report.opportunityCount} calibration opportunities)`
  );
}
