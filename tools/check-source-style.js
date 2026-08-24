#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const GOVERNED_ROOTS = ['src', 'demo'];
const WRITE = process.argv.includes('--write');
const RAW_CONSOLE_ALLOWED_PREFIXES = ['src/cli/', 'src/debug/', 'demo/'];
const RAW_CONSOLE_ALLOWED_FILES = new Set(['src/gpu/device.js']);
const RAW_CONSOLE_PATTERN = /\bconsole\.(?:debug|error|info|log|warn)\s*\(/g;
const INFERRED_GEOMETRY_PATTERNS = [
  {
    pattern: /Math\.sqrt\s*\(\s*(?:elementCount|numElements|numPatches|patchCount)\s*\)/g,
    invariant: 'INV-GEOMETRY-009',
  },
  {
    pattern: /Math\.sqrt\s*\(\s*(?:features|patches|tokens)\.length\s*\)/g,
    invariant: 'INV-GEOMETRY-009',
  },
];

function walkJavaScript(directory, files = []) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const entryPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      walkJavaScript(entryPath, files);
    } else if (entry.name.endsWith('.js')) {
      files.push(entryPath);
    }
  }
  return files;
}

function removeJSDoc(source) {
  return source
    .replace(/^[ \t]*\/\*\*[\s\S]*?\*\/[ \t]*(?:\r?\n)?/gm, '')
    .replace(/\n{3,}/g, '\n\n');
}

const files = GOVERNED_ROOTS.flatMap((root) => walkJavaScript(path.join(ROOT, root)));
const violations = [];
let changedFiles = 0;
let removedBlocks = 0;

for (const file of files) {
  const source = fs.readFileSync(file, 'utf8');
  const relative = path.relative(ROOT, file).split(path.sep).join('/');
  if (WRITE) {
    if (source.includes('/**')) {
      changedFiles += 1;
      removedBlocks += source.match(/\/\*\*/g)?.length ?? 0;
      fs.writeFileSync(file, removeJSDoc(source));
    }
    continue;
  }
  const lines = source.split(/\r?\n/);
  const jsDocLines = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (lines[index].includes('/**')) {
      jsDocLines.push(index + 1);
    }
  }
  if (jsDocLines.length > 0) {
    violations.push({
      file: relative,
      invariant: 'declaration-files-own-api-types',
      detail: `implementation JSDoc at lines ${jsDocLines.join(',')}`,
    });
  }

  const declarationPath = file.replace(/\.js$/u, '.d.ts');
  if (!fs.existsSync(declarationPath)) {
    violations.push({
      file: relative,
      invariant: 'declaration-files-own-api-types',
      detail: 'missing sibling declaration file',
    });
  }

  const rawConsoleAllowed = RAW_CONSOLE_ALLOWED_FILES.has(relative)
    || RAW_CONSOLE_ALLOWED_PREFIXES.some((prefix) => relative.startsWith(prefix));
  if (!rawConsoleAllowed) {
    RAW_CONSOLE_PATTERN.lastIndex = 0;
    const consoleLines = [];
    for (;;) {
      const match = RAW_CONSOLE_PATTERN.exec(source);
      if (!match) break;
      consoleLines.push(source.slice(0, match.index).split(/\r?\n/).length);
    }
    if (consoleLines.length > 0) {
      violations.push({
        file: relative,
        invariant: 'no-raw-runtime-console',
        detail: `raw console call at lines ${consoleLines.join(',')}`,
      });
    }
  }

  for (const check of INFERRED_GEOMETRY_PATTERNS) {
    check.pattern.lastIndex = 0;
    const geometryLines = [];
    for (;;) {
      const match = check.pattern.exec(source);
      if (!match) break;
      geometryLines.push(source.slice(0, match.index).split(/\r?\n/).length);
    }
    if (geometryLines.length > 0) {
      violations.push({
        file: relative,
        invariant: check.invariant,
        detail: `runtime geometry inferred at lines ${geometryLines.join(',')}`,
      });
    }
  }
}

if (WRITE) {
  console.log(
    `[source:style:sync] removed ${removedBlocks} implementation JSDoc block(s) ` +
    `from ${changedFiles} of ${files.length} governed JavaScript modules`
  );
  process.exit(0);
}

if (violations.length === 0) {
  console.log(
    `[source:style:check] ${files.length} governed JavaScript modules have sibling declarations, ` +
    'no implementation JSDoc, no undeclared raw console calls, and no banned geometry inference'
  );
  process.exit(0);
}

console.error(`[source:style:check] ${violations.length} source style violation(s):`);
for (const violation of violations) {
  console.error(`  ${violation.file}: [${violation.invariant}] ${violation.detail}`);
}
process.exit(1);
