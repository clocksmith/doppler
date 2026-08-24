#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const GOVERNED_ROOTS = ['src', 'demo'];
const WRITE = process.argv.includes('--write');

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
  if (!source.includes('/**')) {
    continue;
  }
  if (WRITE) {
    changedFiles += 1;
    removedBlocks += source.match(/\/\*\*/g)?.length ?? 0;
    fs.writeFileSync(file, removeJSDoc(source));
    continue;
  }
  const lines = source.split(/\r?\n/);
  const jsDocLines = [];
  for (let index = 0; index < lines.length; index += 1) {
    if (lines[index].includes('/**')) {
      jsDocLines.push(index + 1);
    }
  }
  violations.push({
    file: path.relative(ROOT, file),
    lines: jsDocLines,
  });
}

if (WRITE) {
  console.log(
    `[source:style:sync] removed ${removedBlocks} implementation JSDoc block(s) ` +
    `from ${changedFiles} of ${files.length} governed JavaScript modules`
  );
  process.exit(0);
}

if (violations.length === 0) {
  console.log(`[source:style:check] ${files.length} governed JavaScript modules contain no JSDoc`);
  process.exit(0);
}

console.error(
  `[source:style:check] ${violations.length} JavaScript module(s) contain JSDoc; ` +
  'move types and API descriptions to the sibling .d.ts file:'
);
for (const violation of violations) {
  console.error(`  ${violation.file}:${violation.lines.join(',')}`);
}
process.exit(1);
