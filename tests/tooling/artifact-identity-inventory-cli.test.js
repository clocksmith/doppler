import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const repoRoot = path.resolve(new URL('../..', import.meta.url).pathname);
const tempDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-artifact-identity-'));

function runInventory(catalogPath, rootPath) {
  return spawnSync(process.execPath, [
    'tools/artifact-identity-inventory.js',
    '--root',
    rootPath,
    '--catalog',
    catalogPath,
    '--check',
    '--json',
  ], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
}

async function writeJson(filePath, payload) {
  await writeFile(filePath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

try {
  const rootPath = path.join(tempDir, 'models', 'local');
  const artifactDir = path.join(rootPath, 'known-incomplete');
  const catalogPath = path.join(tempDir, 'catalog.json');
  await mkdir(artifactDir, { recursive: true });

  await writeJson(path.join(artifactDir, 'manifest.json'), {
    modelId: 'known-incomplete',
    totalSize: 4,
    shards: [
      { index: 0, filename: 'shard_00000.bin', size: 4, offset: 0, hash: 'abcd' },
    ],
  });

  const catalog = {
    models: [
      {
        modelId: 'known-incomplete',
        quickstart: false,
        lifecycle: {
          availability: {
            hf: false,
          },
        },
        sourceCheckpointId: 'unit/source',
        weightPackId: 'known-incomplete-wp',
        manifestVariantId: 'known-incomplete-mv',
        artifactCompleteness: 'incomplete',
        runtimePromotionState: 'unpromoted',
        weightsRefAllowed: false,
      },
    ],
  };

  await writeJson(catalogPath, catalog);
  const unpromoted = runInventory(catalogPath, rootPath);
  assert.equal(unpromoted.status, 0, unpromoted.stderr || unpromoted.stdout);
  const unpromotedReport = JSON.parse(unpromoted.stdout);
  assert.equal(unpromotedReport.summary.errorCount, 0);
  assert.equal(unpromotedReport.summary.warningCount, 2);
  assert.ok(
    unpromotedReport.findings.every((finding) => finding.severity === 'warning'),
    'known unpromoted incomplete artifacts should not be blocking errors'
  );

  catalog.models[0].quickstart = true;
  await writeJson(catalogPath, catalog);
  const quickstart = runInventory(catalogPath, rootPath);
  assert.notEqual(quickstart.status, 0, quickstart.stderr || quickstart.stdout);
  const quickstartReport = JSON.parse(quickstart.stdout);
  assert.ok(
    quickstartReport.findings.some((finding) => finding.severity === 'error'),
    'quickstart incomplete artifacts must remain blocking errors'
  );
} finally {
  await rm(tempDir, { recursive: true, force: true });
}

console.log('artifact-identity-inventory-cli.test: ok');
