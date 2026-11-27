// Same model identity as node-gate.js, without touching a GPU. usage: node identity.js <model-dir>
import fs from 'node:fs'; import path from 'node:path'; import crypto from 'node:crypto';
const dir = process.argv[2]; const sha = (b) => crypto.createHash('sha256').update(b).digest('hex');
const m = fs.readFileSync(path.join(dir, 'manifest.json'), 'utf8');
const files = fs.readdirSync(dir).filter((f) => /^shard_\d+\.bin$|^tokenizer\.json$/.test(f)).sort();
const fd = Object.fromEntries(files.map((f) => [f, sha(fs.readFileSync(path.join(dir, f)))]));
console.log(JSON.stringify({ dir: path.basename(dir), manifestSha256: sha(m), identityDigest: sha(JSON.stringify({ manifest: sha(m), files: fd })), files: fd }));
