// Export the pinned upstream program and data to a scratch directory served by the benchmark.
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { stripTypeScriptTypes } from 'node:module';
import { join, resolve } from 'node:path';

import { tabgptRevision } from './reference.js';
const [checkout, output] = process.argv.slice(2);
if (!checkout || !output) {
  throw new Error('Usage: node prepare-reference.js <jbingen/tabgpt checkout> <served scratch directory>');
}

function readPinned(path) {
  return execFileSync('git', ['-C', resolve(checkout), 'show', tabgptRevision + ':' + path], {
    maxBuffer: 12 * 1024 * 1024,
  });
}

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

await mkdir(output, { recursive: true });
const modules = {};
for (const name of ['kernels', 'data', 'model']) {
  const path = 'src/engine/' + name + '.ts';
  const source = readPinned(path);
  // The browser's ESM loader requires the extension on TabGPT's relative import.
  const javascript = stripTypeScriptTypes(source.toString('utf8')).replace("from './kernels'", "from './kernels.js'");
  await writeFile(join(output, name + '.js'), javascript);
  modules[name] = { path, sourceSha256: sha256(source), javascriptSha256: sha256(javascript) };
}
const metadata = JSON.parse(readPinned('public/tokens-nob-meta.json'));
const tokenizer = readPinned('public/' + metadata.tok);
const vocabSize = JSON.parse(tokenizer).vocab.length;
const splits = {};
for (const [name, filename] of Object.entries({ train: metadata.shards[0], val: metadata.val })) {
  const path = 'public/' + filename;
  const data = readPinned(path);
  await writeFile(join(output, name + '.bin'), data);
  splits[name] = { path, sha256: sha256(data), tokens: data.length / Uint16Array.BYTES_PER_ELEMENT };
}
const manifest = {
  repository: 'https://github.com/jbingen/tabgpt',
  revision: tabgptRevision,
  tokenizer: { path: metadata.tok, sha256: sha256(tokenizer), vocabSize },
  modules,
  splits,
};
await writeFile(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
