#!/usr/bin/env node
// Reproduce the retained Atlas without a models drive or a scratch directory.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const source = path.join(root, 'tools/model-atlas');
const args = process.argv.slice(2);
const outputFlag = args.indexOf('--output');
if (outputFlag < 0 || !args[outputFlag + 1] || args.some((arg, i) => !['--output', '--check'].includes(arg) && i !== outputFlag + 1)) {
  throw new Error('Usage: node tools/build-model-atlas.js --output /path/to/index.html [--check] (requires python3)');
}
const output = path.resolve(args[outputFlag + 1]);
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'doppler-atlas-'));
try {
  const provenance = JSON.parse(fs.readFileSync(path.join(source, 'snapshot.json'), 'utf8'));
  for (const [file, expected] of Object.entries(provenance.inputs)) {
    const bytes = fs.readFileSync(path.join(source, file));
    if (createHash('sha256').update(bytes).digest('hex') !== expected) throw new Error(`Atlas snapshot digest mismatch: ${file}`);
    fs.writeFileSync(path.join(temporary, file), bytes);
  }
  // The relocated generator retains its original Python rendering logic. It reads
  // current catalog lifecycle fields, while shape counts remain a labeled snapshot.
  execFileSync('python3', [path.join(temporary, 'gen.py'), temporary, root], { stdio: 'pipe' });
  let page = fs.readFileSync(path.join(temporary, 'template.html'), 'utf8')
    .replace('{{MATRIX}}', fs.readFileSync(path.join(temporary, 'matrix.html'), 'utf8'))
    .replace('{{CARDS}}', fs.readFileSync(path.join(temporary, 'cards.html'), 'utf8'))
    .replace('<meta name="claude:viewport-fit-sides" content="cover">\n', '')
    .replace('</style>', 'html{-webkit-text-size-adjust:100%}body{margin:0}</style>');
  if (/\{\{(?:MATRIX|CARDS)\}\}/.test(page)) throw new Error('Atlas contains an unresolved template slot');
  const parts = page.split('</style>');
  if (parts.length !== 2) throw new Error('Atlas template must have one style block');
  page = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n<meta name="description" content="Doppler model formats, loading paths and pipelines, with recorded precision and explicit qualification limits.">\n<meta name="color-scheme" content="light dark">\n<link rel="icon" type="image/svg+xml" href="/doppler/demo/favicon.svg">\n' + parts[0] + '</style>\n</head>\n<body>\n' + parts[1].trim() + '\n</body>\n</html>\n';
  if (/\b(jupiter|mars)\b|\/Users\/|\/home\/|\/run\/media\/|d4da-analytics/i.test(page)) throw new Error('Atlas contains a private identifier or analytics');
  if (args.includes('--check')) {
    if (!fs.existsSync(output) || fs.readFileSync(output, 'utf8') !== page) throw new Error(`Atlas is stale: ${output}`);
  } else {
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output + '.tmp', page);
    fs.renameSync(output + '.tmp', output);
  }
  console.log(JSON.stringify({ atlas: args.includes('--check') ? 'verified' : 'generated', bytes: Buffer.byteLength(page), sha256: createHash('sha256').update(page).digest('hex') }));
} finally {
  fs.rmSync(temporary, { recursive: true, force: true });
}
