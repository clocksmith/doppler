import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const tracked = new Set(execFileSync('git', ['ls-files', '-z'], { cwd: repo, encoding: 'utf8' }).split('\0'));
let packets = 0;
for (const directory of readdirSync(join(repo, 'artifacts')).filter(name => name.startsWith('model-repairs-'))) {
  for (const packet of readdirSync(join(repo, 'artifacts', directory), { withFileTypes: true })) {
    if (!packet.isDirectory()) continue;
    const root = join(repo, 'artifacts', directory, packet.name);
    const receipt = JSON.parse(readFileSync(join(root, 'receipt.json'), 'utf8'));
    assert.ok(receipt.files.length > 0, `${root}: receipt must name its evidence files`);
    const seen = new Set();
    for (const entry of receipt.files) {
      assert.equal(typeof entry.path, 'string');
      assert.ok(!entry.path.startsWith('/') && !entry.path.split(/[\\/]/).includes('..'));
      assert.ok(!seen.has(entry.path), `${root}: duplicate member ${entry.path}`); seen.add(entry.path);
      const filename = join(root, entry.path);
      const repoPath = relative(repo, filename).split('\\').join('/');
      assert.ok(tracked.has(repoPath), `Receipt member must be tracked: ${repoPath}`);
      const bytes = readFileSync(filename);
      assert.equal(bytes.length, entry.bytes, `${filename}: size mismatch`);
      assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256, `${filename}: digest mismatch`);
    }
    packets++;
  }
}
assert.ok(packets >= 4, 'Repair and publication packets must be present.');
console.log(`model-repair-receipts: ${packets} complete tracked packets`);
