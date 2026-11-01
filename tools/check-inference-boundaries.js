import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const attentionRoots = [
  'src/inference/pipelines/text/attention',
  'src/inference/pipelines/text/linear-attention',
];
const attentionFiles = ['src/inference/pipelines/text/linear-attention-ab-fusion.js'];
const violations = [];

async function visit(directory) {
  const entries = await fs.readdir(directory, { withFileTypes: true });
  for (const entry of entries) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      await visit(entryPath);
      continue;
    }
    if (!entry.isFile() || !entry.name.endsWith('.js')) {
      continue;
    }
    await inspectFile(entryPath);
  }
}

async function inspectFile(entryPath) {
    const source = await fs.readFile(entryPath, 'utf8');
    const lines = source.split(/\r?\n/);
    for (let index = 0; index < lines.length; index += 1) {
      if (/\bgetRuntimeConfig\s*\(/.test(lines[index])) {
        violations.push({
          file: path.relative(repoRoot, entryPath),
          line: index + 1,
          source: lines[index].trim(),
        });
      }
    }
}

for (const root of attentionRoots) await visit(path.join(repoRoot, root));
for (const file of attentionFiles) await inspectFile(path.join(repoRoot, file));

if (violations.length > 0) {
  console.error('Inference boundary check failed: deep attention runtime-config reads are forbidden.');
  for (const violation of violations) {
    console.error(`${violation.file}:${violation.line}: ${violation.source}`);
  }
  process.exitCode = 1;
} else {
  console.log('inference boundary check passed');
  console.log('- INV-ATTN-003: no deep runtime-config reads in attention or linear projection fusion');
}
