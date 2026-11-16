import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { readTrackedArtifactInventory, summarizeArtifactInventory, validateArtifactRetention } from '../../tools/check-artifact-retention.js';

const root = fs.mkdtempSync(path.join(os.tmpdir(), 'doppler-retention-test-'));
try {
  const git = args => execFileSync('git', args, { cwd: root, encoding: 'utf8' });
  git(['init', '--quiet']);
  fs.mkdirSync(path.join(root, 'reports/run'), { recursive: true });
  fs.writeFileSync(path.join(root, 'reports/run/result.log'), 'frozen bytes\n');
  git(['add', 'reports']);
  const inventory = readTrackedArtifactInventory(root);
  const originalObjects = git(['count-objects', '-v']);
  const summaries = summarizeArtifactInventory(inventory);
  const policy = {
    schemaVersion: 1,
    retentionClasses: { frozen: { owner: 'test-owner', reason: 'Pinned regression receipt', removalCondition: 'Replacement evidence retires the fixture' } },
    campaigns: Object.fromEntries(Object.entries(summaries).map(([key, value]) => [key, { retentionClass: 'frozen', ...value }])),
  };
  assert.deepEqual(validateArtifactRetention(policy, inventory), []);
  fs.writeFileSync(path.join(root, 'reports/run/result.log'), 'changed bytes\n');
  assert.match(validateArtifactRetention(policy, readTrackedArtifactInventory(root)).join(';'), /inventory changed/);
  assert.equal(git(['count-objects', '-v']), originalObjects, 'checking unstaged evidence must not write Git objects');
  assert.match(validateArtifactRetention(policy, [...inventory, { path: 'reports/new/result.json', objectId: 'new', bytes: 5 }]).join(';'), /no retention record/);
  assert.match(validateArtifactRetention(policy, []).join(';'), /stale retention record/);
  const expired = structuredClone(policy);
  expired.retentionClasses.frozen.retainUntil = '2026-01-01';
  assert.match(validateArtifactRetention(expired, inventory, '2026-11-15').join(';'), /expired/);
  expired.retentionClasses.frozen.retainUntil = '2026-02-31';
  assert.match(validateArtifactRetention(expired, inventory, '2026-11-15').join(';'), /invalid retainUntil/);
  const unowned = structuredClone(policy);
  delete unowned.retentionClasses.frozen.owner;
  assert.match(validateArtifactRetention(unowned, inventory).join(';'), /requires owner/);
} finally {
  fs.rmSync(root, { recursive: true, force: true });
}
console.log('artifact retention: unowned, modified, missing and expired evidence fails without mutating Git');
