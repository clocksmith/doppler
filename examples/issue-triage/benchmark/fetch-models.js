#!/usr/bin/env node
// Fetch the benchmark's pinned Hugging Face models; verify every LFS file against
// the tree's sha256 oid.
//   node fetch-models.js <destination root>
// Writes <root>/<repo id>/ (the layout Transformers.js reads) plus acquisition.json
// per repository.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

export const PINNED = [
  ['Qwen/Qwen3-Reranker-0.6B', 'e61197ed45024b0ed8a2d74b80b4d909f1255473', []],
  ['onnx-community/kev-0.6b-ONNX', '5a827091b3d58fe4fdfc02a5a5b2ce364047d678', []],
  ['onnx-community/kev-4b-ONNX', '195bdf73096df2bca7150223c9279af7893163c4', []],
  ['onnx-community/open-jev-deberta-v3-large-ONNX', '7c79f25b5ac496089f448a969c801872ad59d31c', ['onnx/model.onnx']],
];

class FetchError extends Error {}

async function request(url) {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) throw new FetchError(`HTTP ${response.status} for ${url}`);
  return response;
}

export async function sha256File(file) {
  const hash = createHash('sha256');
  await pipeline(fs.createReadStream(file, { highWaterMark: 1 << 22 }), hash);
  return hash.digest('hex');
}

// Files of a tree listing that should be fetched, after the skip prefixes.
export function selectFiles(tree, skip) {
  return tree.filter((entry) => entry.type === 'file' && !skip.some((prefix) => entry.path.startsWith(prefix)));
}

// acquisition.json text: one-space indent, non-ASCII escaped, no trailing newline.
export function acquisitionText(record) {
  return JSON.stringify(record, null, 1).replace(/[\u007f-￿]/g, (ch) => '\\u' + ch.charCodeAt(0).toString(16).padStart(4, '0'));
}

export async function fetchRepository(repo, rev, out, skip) {
  const tree = await (await request(`https://huggingface.co/api/models/${repo}/tree/${rev}?recursive=true`)).json();
  const record = [];
  for (const entry of selectFiles(tree, skip)) {
    const dest = path.join(out, entry.path);
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    const oid = entry.lfs?.oid || null;
    if (!(fs.existsSync(dest) && fs.statSync(dest).size === entry.size)) {
      const response = await request(`https://huggingface.co/${repo}/resolve/${rev}/${entry.path}`);
      await pipeline(Readable.fromWeb(response.body), fs.createWriteStream(dest + '.part'));
      fs.renameSync(dest + '.part', dest);
    }
    const digest = await sha256File(dest);
    if (oid && digest !== oid) throw new FetchError(`sha256 mismatch for ${repo}/${entry.path}: ${digest} != ${oid}`);
    if (fs.statSync(dest).size !== entry.size) throw new FetchError(`size mismatch for ${repo}/${entry.path}`);
    record.push({ path: entry.path, size: entry.size, sha256: digest, lfsOidVerified: Boolean(oid) });
  }
  fs.writeFileSync(path.join(out, 'acquisition.json'), acquisitionText({ repository: repo, revision: rev, files: record }));
  console.log(repo, record.length, 'files ok');
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = process.argv[2];
  if (!root) {
    console.error('Usage: node fetch-models.js <destination root>');
    process.exit(2);
  }
  try {
    for (const [repo, rev, skip] of PINNED) await fetchRepository(repo, rev, path.join(root, repo), skip);
  } catch (error) {
    console.error(error instanceof FetchError ? error.message : error);
    process.exit(1);
  }
}
