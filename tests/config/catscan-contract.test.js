import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { buildCatscanReport } from '../../tools/sync-catscan-index.js';

const FREEDOM = 'Any implementation is permitted if it preserves these boundaries and passes the acceptance evidence.';

function charter({ component, parent, target }) {
  return `# CATSCAN: Fixture

Component: \`${component}\`

Parent: ${parent}

## Target

${target}

## Authority

- Owns fixture behavior.
- Does not own neighboring fixture behavior.

## Scope

- The fixture component.

## Contracts

- Input: [Fixture evidence](evidence.txt).
- Output: A validated fixture.

## Invariants

- Fixture failures remain explicit.

## Acceptance

- The fixture validates.
- Evidence: [Fixture evidence](evidence.txt).

## Non-goals

- Production behavior.

## Freedom

${FREEDOM}
`;
}

const repositoryReport = await buildCatscanReport({ repoRoot: process.cwd() });
assert.equal(repositoryReport.ok, true, repositoryReport.errors.join('\n'));
assert.equal(repositoryReport.records.length, 28);

const fixtureRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-catscan-'));
try {
  await fs.mkdir(path.join(fixtureRoot, 'src'), { recursive: true });
  await fs.mkdir(path.join(fixtureRoot, 'tools', 'policies'), { recursive: true });
  await fs.mkdir(path.join(fixtureRoot, 'docs'), { recursive: true });
  await fs.writeFile(path.join(fixtureRoot, 'evidence.txt'), 'fixture evidence\n', 'utf8');
  await fs.writeFile(path.join(fixtureRoot, 'src', 'evidence.txt'), 'child evidence\n', 'utf8');

  const policy = {
    $schema: '../../src/config/schema/catscan-policy.schema.json',
    schemaVersion: 1,
    source: 'doppler',
    charterFilename: 'CATSCAN.md',
    indexPath: 'docs/component-index.md',
    maxWords: 250,
    requiredMetadata: ['Component', 'Parent'],
    requiredSections: [
      'Target',
      'Authority',
      'Scope',
      'Contracts',
      'Invariants',
      'Acceptance',
      'Non-goals',
      'Freedom',
    ],
    freedomText: FREEDOM,
    ignoredDirectories: ['node_modules'],
    requiredCharterPaths: ['CATSCAN.md', 'src/CATSCAN.md'],
  };
  const policyPath = path.join(fixtureRoot, 'tools', 'policies', 'catscan-policy.json');
  const rootSource = charter({
    component: 'doppler',
    parent: 'none',
    target: 'Validate the root fixture.',
  });
  const childSource = charter({
    component: 'doppler.child',
    parent: '[Fixture](../CATSCAN.md)',
    target: 'Validate the child fixture.',
  });
  await fs.writeFile(policyPath, `${JSON.stringify(policy, null, 2)}\n`, 'utf8');
  await fs.writeFile(path.join(fixtureRoot, 'CATSCAN.md'), rootSource, 'utf8');
  await fs.writeFile(path.join(fixtureRoot, 'src', 'CATSCAN.md'), childSource, 'utf8');

  const stale = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath });
  assert.equal(stale.ok, false);
  assert.match(stale.errors.join('\n'), /component-index\.md is stale/);

  const valid = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.equal(valid.ok, true, valid.errors.join('\n'));
  await fs.writeFile(valid.indexPath, valid.renderedIndex, 'utf8');
  const current = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath });
  assert.equal(current.ok, true, current.errors.join('\n'));

  await fs.writeFile(
    path.join(fixtureRoot, 'src', 'CATSCAN.md'),
    childSource.replace('`doppler.child`', '`doppler`'),
    'utf8'
  );
  const duplicate = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.match(duplicate.errors.join('\n'), /duplicate Component ID doppler/);

  await fs.writeFile(
    path.join(fixtureRoot, 'src', 'CATSCAN.md'),
    childSource.replace('(../CATSCAN.md)', '(evidence.txt)'),
    'utf8'
  );
  const wrongParent = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.match(wrongParent.errors.join('\n'), /Parent must resolve to nearest ancestor CATSCAN\.md/);

  await fs.writeFile(
    path.join(fixtureRoot, 'src', 'CATSCAN.md'),
    childSource.replace(/## Scope\n\n- The fixture component\.\n\n/, ''),
    'utf8'
  );
  const missingSection = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.match(missingSection.errors.join('\n'), /missing section "Scope"/);

  await fs.writeFile(
    path.join(fixtureRoot, 'src', 'CATSCAN.md'),
    childSource.replaceAll('(evidence.txt)', '(missing.txt)'),
    'utf8'
  );
  const missingLink = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.match(missingLink.errors.join('\n'), /link target does not exist: src\/missing\.txt/);

  policy.maxWords = 100;
  await fs.writeFile(policyPath, `${JSON.stringify(policy, null, 2)}\n`, 'utf8');
  await fs.writeFile(
    path.join(fixtureRoot, 'src', 'CATSCAN.md'),
    childSource.replace('Validate the child fixture.', `${'word '.repeat(120)}.`),
    'utf8'
  );
  const oversized = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.match(oversized.errors.join('\n'), /words exceeds the 100-word limit/);

  policy.maxWords = 250;
  policy.requiredCharterPaths.push('missing/CATSCAN.md');
  await fs.writeFile(policyPath, `${JSON.stringify(policy, null, 2)}\n`, 'utf8');
  await fs.writeFile(path.join(fixtureRoot, 'src', 'CATSCAN.md'), childSource, 'utf8');
  const missingInventory = await buildCatscanReport({ repoRoot: fixtureRoot, policyPath, checkIndex: false });
  assert.match(missingInventory.errors.join('\n'), /missing required charter: missing\/CATSCAN\.md/);
} finally {
  await fs.rm(fixtureRoot, { recursive: true, force: true });
}

console.log('catscan-contract.test: ok');
