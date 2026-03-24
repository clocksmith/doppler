#!/usr/bin/env node

/**
 * Regenerates src/config/kernels/kernel-ref-digests.js from
 * src/config/kernels/registry.json + src/gpu/kernels/*.wgsl.
 *
 * Usage:
 *   node tools/sync-kernel-ref-digests.js          # write
 *   node tools/sync-kernel-ref-digests.js --check   # exit 1 if stale
 */

import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256Hex } from '../src/utils/sha256.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..');
const registryPath = path.join(repoRoot, 'src/config/kernels/registry.json');
const kernelsDir = path.join(repoRoot, 'src/gpu/kernels');
const outputPath = path.join(repoRoot, 'src/config/kernels/kernel-ref-digests.js');

const checkMode = process.argv.includes('--check');

const registry = JSON.parse(await fs.readFile(registryPath, 'utf8'));
const digests = new Map();

for (const opSchema of Object.values(registry.operations ?? {})) {
  for (const variant of Object.values(opSchema.variants ?? {})) {
    const wgsl = variant?.wgsl;
    const entry = variant?.entryPoint ?? 'main';
    if (typeof wgsl !== 'string' || wgsl.length === 0) {
      continue;
    }
    const key = `${wgsl}#${entry}`;
    if (digests.has(key)) {
      continue;
    }
    const shaderPath = path.join(kernelsDir, wgsl);
    const shaderSource = (await fs.readFile(shaderPath, 'utf8')).replace(/\r\n/g, '\n');
    digests.set(key, sha256Hex(`${shaderSource}\n@@entry:${entry}`));
  }
}

const sorted = [...digests.entries()].sort((a, b) => a[0].localeCompare(b[0]));
const lines = sorted.map(([key, hash]) => `  ${JSON.stringify(key)}: ${JSON.stringify(hash)},`);
const content = `// Auto-generated from src/config/kernels/registry.json + src/gpu/kernels/*.wgsl
// Content hash input: normalized WGSL source + "\\n@@entry:<entry>"
export const KERNEL_REF_CONTENT_DIGESTS = Object.freeze({
${lines.join('\n')}
});
`;

if (checkMode) {
  const existing = await fs.readFile(outputPath, 'utf8');
  if (existing === content) {
    console.log('[digests:check] kernel-ref-digests.js is up to date');
    process.exit(0);
  }
  console.error('[digests:check] kernel-ref-digests.js is stale — run: npm run digests:sync');
  process.exit(1);
}

await fs.writeFile(outputPath, content);
console.log(`[digests:sync] wrote ${sorted.length} entries to src/config/kernels/kernel-ref-digests.js`);
