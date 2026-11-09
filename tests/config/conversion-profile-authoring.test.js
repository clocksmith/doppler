import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { syncConversionProfiles, expandReferences, renderProfile, AUTHORING_ROOT } from '../../tools/sync-conversion-profiles.js';

// The committed profiles are exactly what their authoring sources expand to.
const rows = syncConversionProfiles();
assert.ok(rows.length >= 4, 'the four shared-execution Qwen profiles are authored');
assert.deepEqual(rows.filter((row) => row.status !== 'current'), [], 'every authored profile is current');
const targets = rows.map((row) => row.target);
for (const name of ['qwen-3-5-2b-q4k-ehaf16', 'qwen-3-5-9b-q4k-ehaf16', 'qwen-3-6-27b-q4k-eaf16', 'qwen-3-6-27b-q4k-ehaf16']) {
  assert.ok(targets.includes(`src/config/conversion/qwen3/${name}.json`), name);
}

// Model-specific metadata stays per profile; only the shared execution is referenced.
const authoring = JSON.parse(fs.readFileSync(new URL(`../../${AUTHORING_ROOT}/qwen3/qwen-3-5-2b-q4k-ehaf16.authoring.json`, import.meta.url), 'utf8'));
assert.deepEqual(authoring.profile.execution, { $ref: 'shared/q4k-ehaf16-execution.json' });
assert.equal(typeof authoring.profile.manifest, 'object');
assert.equal(JSON.stringify(authoring.profile.manifest).includes('$ref'), false);

// Drift is detected and repaired in a scratch copy of the repository layout.
const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'doppler-profile-authoring-'));
try {
  fs.cpSync(new URL(`../../${AUTHORING_ROOT}`, import.meta.url), path.join(scratch, AUTHORING_ROOT), { recursive: true });
  for (const target of targets) {
    fs.mkdirSync(path.dirname(path.join(scratch, target)), { recursive: true });
    fs.copyFileSync(new URL(`../../${target}`, import.meta.url), path.join(scratch, target));
  }
  const victim = path.join(scratch, targets[0]);
  const profile = JSON.parse(fs.readFileSync(victim, 'utf8'));
  profile.execution.drifted = true;
  fs.writeFileSync(victim, renderProfile(profile));
  fs.rmSync(path.join(scratch, targets[1]));
  const checked = syncConversionProfiles({ root: scratch });
  assert.equal(checked.find((row) => row.target === targets[0]).status, 'drifted');
  assert.equal(checked.find((row) => row.target === targets[1]).status, 'missing');
  syncConversionProfiles({ root: scratch, write: true });
  assert.deepEqual(syncConversionProfiles({ root: scratch }).map((row) => row.status), rows.map(() => 'current'));
  assert.equal(fs.readFileSync(victim, 'utf8'), fs.readFileSync(new URL(`../../${targets[0]}`, import.meta.url), 'utf8'), 'sync restores the byte-identical profile');

  // References cannot escape the authoring root or cycle.
  assert.throws(() => expandReferences({ $ref: '../../../package.json' }, path.join(scratch, AUTHORING_ROOT, 'qwen3'), scratch), /leaves tools\/conversion-authoring/);
  const loop = path.join(scratch, AUTHORING_ROOT, 'loop.json');
  fs.writeFileSync(loop, JSON.stringify({ $ref: 'loop.json' }));
  assert.throws(() => expandReferences({ $ref: 'loop.json' }, path.dirname(loop), scratch), /cycle/);
  assert.throws(() => expandReferences({ $ref: 'loop.json', extra: 1 }, path.dirname(loop), scratch), /carries only \$ref/);
} finally {
  fs.rmSync(scratch, { recursive: true, force: true });
}

console.log('conversion-profile-authoring: shared execution expands to byte-identical profiles and drift is detected');
