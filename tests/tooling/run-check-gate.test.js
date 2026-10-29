import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { readFileSync } from 'node:fs';

import {
  POLICY_PATH,
  globToRegExp,
  localImportClosure,
  referencedPaths,
  scriptEntryFiles,
  selectTriggeredChecks,
  validatePolicy,
} from '../../tools/run-check-gate.js';

const { scripts } = JSON.parse(readFileSync('package.json', 'utf8'));
const policy = JSON.parse(readFileSync(POLICY_PATH, 'utf8'));
const selected = (files) => selectTriggeredChecks(policy, scripts, files).map((item) => item.name);

// The checked-in policy names only declared scripts and partitions them.
assert.deepEqual(validatePolicy(policy, scripts), []);
for (const name of policy.behavior) assert.ok(scripts[name], `${name} must be an npm script`);

// Glob semantics: ** crosses directories, * does not.
assert.ok(globToRegExp('**/CATSCAN.md').test('CATSCAN.md'));
assert.ok(globToRegExp('**/CATSCAN.md').test('src/gpu/CATSCAN.md'));
assert.ok(globToRegExp('src/*.d.ts').test('src/index.d.ts'));
assert.equal(globToRegExp('src/*.d.ts').test('src/gpu/device.d.ts'), false);
assert.equal(globToRegExp('src/rules/**').test('src/rulesx/a.json'), false);

// Malformed policies fail before any check runs.
assert.match(validatePolicy({ schemaVersion: 1, behavior: ['missing:script'], triggered: {} }, scripts).join('\n'),
  /missing:script" is not declared/);
assert.match(validatePolicy({ schemaVersion: 1, behavior: ['test:unit'], triggered: { 'test:unit': { inputs: ['x'] } } }, scripts).join('\n'),
  /both behavior and triggered/);
assert.match(validatePolicy({ schemaVersion: 1, behavior: ['test:unit'], triggered: { 'catscan:check': { inputs: [] } } }, scripts).join('\n'),
  /non-empty array/);
assert.match(validatePolicy({ schemaVersion: 1, behavior: ['test:unit'], triggered: { 'catscan:check': { inputs: ['x'], extra: 1 } } }, scripts).join('\n'),
  /unknown fields extra/);

// A script's entry files include nested npm scripts and listed test files.
const releaseEntries = scriptEntryFiles('model-release:check', scripts);
assert.ok(releaseEntries.includes('tools/check-model-release-platform.js'));
assert.ok(releaseEntries.includes('tests/config/model-release-platform.test.js'));
assert.ok(releaseEntries.includes('tests/tooling/production-release.test.js'), 'nested production-release:check is expanded');

// Selection follows declared globs.
assert.ok(selected(['src/gpu/CATSCAN.md']).includes('catscan:check'));
assert.ok(selected(['src/rules/inference/capability-transforms.rules.json']).includes('rules:bundle:check'));
assert.ok(selected(['src/rules/inference/capability-transforms.rules.json']).includes('capability-policy:check'));
assert.equal(selected(['src/rules/inference/capability-transforms.rules.json']).includes('catscan:check'), false);

// Selection follows each check's own code: the tool and its local imports.
assert.ok(selected(['tools/sync-catscan-index.js']).includes('catscan:check'));
const catscanClosure = localImportClosure(scriptEntryFiles('catscan:check', scripts));
for (const file of catscanClosure) {
  assert.ok(selected([file]).includes('catscan:check'), `${file} is in the catscan:check closure`);
}

// Selection follows evidence named by a referencedBy JSON file.
const promotionReferences = [...referencedPaths('tools/policies/runtime-promotion-monitoring.json')]
  .filter((file) => file !== 'tools/policies/runtime-promotion-monitoring.json' && !file.startsWith('src/') && !file.startsWith('tools/'));
for (const file of promotionReferences) {
  assert.ok(selected([file]).includes('promotion:monitoring:check'), `${file} is promotion evidence`);
}

// Unrelated prose selects nothing; gate changes select every triggered check.
assert.deepEqual(selected(['LICENSE']), []);
assert.deepEqual(selected([POLICY_PATH]).sort(), Object.keys(policy.triggered).sort());
assert.deepEqual(selected(['tools/run-check-gate.js']).sort(), Object.keys(policy.triggered).sort());

// referencedPaths only returns files that exist, and a missing JSON file fails.
{
  const root = mkdtempSync(path.join(tmpdir(), 'doppler-check-gate-'));
  try {
    mkdirSync(path.join(root, 'evidence'));
    writeFileSync(path.join(root, 'evidence/receipt.json'), '{}');
    writeFileSync(path.join(root, 'policy.json'), JSON.stringify({
      present: 'evidence/receipt.json', absent: 'evidence/missing.json', prose: 'not a path',
    }));
    assert.deepEqual([...referencedPaths('policy.json', root)].sort(), ['evidence/receipt.json', 'policy.json']);
    assert.throws(() => referencedPaths('missing.json', root), /referencedBy file not found/);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
}

// The CLI rejects ambiguous or empty selections.
for (const args of [[], ['--all', '--changed'], ['--base'], ['--unknown']]) {
  const result = spawnSync(process.execPath, ['tools/run-check-gate.js', ...args], { encoding: 'utf8' });
  assert.notEqual(result.status, 0, `run-check-gate ${args.join(' ')} must fail`);
}
{
  const result = spawnSync(process.execPath, ['tools/run-check-gate.js', '--behavior', '--list'], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual(JSON.parse(result.stdout).behavior, policy.behavior);
}

console.log('run-check-gate.test: ok');
