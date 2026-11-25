// Test-only CPU stand-in for shares_evaluator_gpu.js: the same pipe protocol and manifest, with the product computed
// by the client's own fieldProduct. `--fault N` adds one to output element N of every answer, as a dishonest provider.
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { POLICY, PRIME, fieldProduct } from '../../../tools/privacy-concrete/shares_client.js';

const argument = (name, fallback) => {
  const index = process.argv.indexOf(name);
  return index < 0 ? fallback : process.argv[index + 1];
};
const directory = argument('--cache-dir');
const fault = Number(argument('--fault', '-1'));
const manifest = JSON.parse(readFileSync(join(directory, 'shares-manifest.json'), 'utf8'));
const projections = new Map(manifest.projections.map((row) => [row.name, row]));
const matrices = new Map();
function matrix(row) {
  if (!matrices.has(row.name)) {
    const bytes = readFileSync(join(directory, row.file));
    if (createHash('sha256').update(bytes).digest('hex') !== row.weightId) throw new Error('Public coefficient identity changed');
    const copy = Buffer.from(bytes);
    matrices.set(row.name, new Int16Array(copy.buffer, copy.byteOffset, row.inputWidth * row.outputWidth));
  }
  return matrices.get(row.name);
}

const chunks = [];
let buffered = 0, ended = false, wake = null;
process.stdin.on('data', (chunk) => { chunks.push(chunk); buffered += chunk.length; wake?.(); });
process.stdin.on('end', () => { ended = true; wake?.(); });
async function read(length) {
  while (buffered < length) {
    if (ended) return null;
    await new Promise((resolve) => { wake = resolve; });
    wake = null;
  }
  const all = Buffer.concat(chunks);
  chunks.length = 0;
  if (all.length > length) chunks.push(all.subarray(length));
  buffered -= length;
  return all.subarray(0, length);
}
function respond(sequence, status, rows, width, payload) {
  const head = Buffer.alloc(21);
  head.write('SHR1', 0, 'latin1');
  head.writeBigUInt64LE(sequence, 4);
  head.writeUInt8(status, 12);
  head.writeUInt32LE(rows, 13);
  head.writeUInt32LE(width, 17);
  process.stdout.write(payload ? Buffer.concat([head, payload]) : head);
}

for (;;) {
  const magic = await read(4);
  if (magic === null || magic.toString('latin1') === 'SHC1') break;
  const head = await read(18);
  const sequence = head.readBigUInt64LE(0), nameBytes = head.readUInt16LE(8), rows = head.readUInt32LE(10), width = head.readUInt32LE(14);
  const row = projections.get((await read(nameBytes)).toString('utf8'));
  if (magic.toString('latin1') !== 'SHQ1' || !row || width !== row.inputWidth || rows < 1
    || rows > (1 + 2 * POLICY.checkRounds) * POLICY.maximumRows) {
    respond(sequence, 1, 0, 0);
    break;
  }
  const shares = new BigUint64Array(rows * width);
  new Uint8Array(shares.buffer).set(await read(8 * rows * width));
  const answer = fieldProduct(shares, rows, matrix(row), row.inputWidth, row.outputWidth);
  if (fault >= 0 && fault < answer.length) answer[fault] = (answer[fault] + 1n) % PRIME;
  respond(sequence, 0, rows, row.outputWidth, Buffer.from(answer.buffer));
}
await new Promise((resolve) => process.stdout.write('', resolve));
process.exit(0);
