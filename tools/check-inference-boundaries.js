import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const attentionRoot = path.join(repoRoot, 'src/inference/pipelines/text/attention');
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
}

await visit(attentionRoot);

if (violations.length > 0) {
  console.error('Inference boundary check failed: deep attention runtime-config reads are forbidden.');
  for (const violation of violations) {
    console.error(`${violation.file}:${violation.line}: ${violation.source}`);
  }
  process.exitCode = 1;
} else {
  console.log('inference boundary check passed');
  console.log('- INV-ATTN-003: no getRuntimeConfig() calls under attention/');
}
