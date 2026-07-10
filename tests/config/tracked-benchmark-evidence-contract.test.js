import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

const repoRoot = process.cwd();
const resultsDir = path.join(repoRoot, 'benchmarks', 'vendors', 'results');
const untrackedReceiptPath = path.join(
  resultsDir,
  `embedding_compare_untracked-ci-fixture-${process.pid}.json`
);
const outputDir = await mkdtemp(path.join(os.tmpdir(), 'doppler-tracked-evidence-'));

function runGenerator(script, jsonOutput, markdownOutput) {
  const result = spawnSync(
    process.execPath,
    [script, '--json-output', jsonOutput, '--markdown-output', markdownOutput],
    {
      cwd: repoRoot,
      encoding: 'utf8',
    }
  );
  assert.equal(
    result.status,
    0,
    `${script} failed\nstdout:\n${result.stdout}\nstderr:\n${result.stderr}`
  );
}

try {
  await writeFile(untrackedReceiptPath, `${JSON.stringify({
    schemaVersion: 1,
    kind: 'embedding-engine-compare',
    timestamp: '2999-01-01T00:00:00.000Z',
    model: {
      dopplerModelId: 'google-embeddinggemma-300m-q4k-ehf16-af32',
    },
    compareLane: {
      claimable: true,
      correctnessOk: true,
    },
    summary: {
      correctnessOk: true,
      doppler: {
        speed: {
          medianEmbeddingMs: 1,
          avgEmbeddingsPerSec: 1000,
          p95EmbeddingMs: 1,
          modelLoadMs: 1,
          embeddingDim: 768,
        },
      },
      transformersjs: {
        speed: {
          medianEmbeddingMs: 1000,
          avgEmbeddingsPerSec: 1,
          p95EmbeddingMs: 1000,
          modelLoadMs: 1000,
          embeddingDim: 768,
        },
      },
    },
  }, null, 2)}\n`, 'utf8');

  const inventoryJson = path.join(outputDir, 'inventory.json');
  const inventoryMarkdown = path.join(outputDir, 'inventory.md');
  runGenerator('tools/sync-model-support-inventory.js', inventoryJson, inventoryMarkdown);

  const scoreboardJson = path.join(outputDir, 'scoreboard.json');
  const scoreboardMarkdown = path.join(outputDir, 'scoreboard.md');
  runGenerator('tools/sync-model-competition-scoreboard.js', scoreboardJson, scoreboardMarkdown);

  assert.deepEqual(
    JSON.parse(await readFile(inventoryJson, 'utf8')),
    JSON.parse(await readFile('benchmarks/vendors/model-support-inventory.json', 'utf8')),
    'support inventory must be derived only from tracked benchmark receipts'
  );
  assert.equal(
    await readFile(inventoryMarkdown, 'utf8'),
    await readFile('docs/model-support-inventory.md', 'utf8'),
    'support inventory Markdown must ignore untracked benchmark receipts'
  );
  assert.deepEqual(
    JSON.parse(await readFile(scoreboardJson, 'utf8')),
    JSON.parse(await readFile('benchmarks/vendors/model-competition-scoreboard.json', 'utf8')),
    'competition scoreboard must be derived only from tracked benchmark receipts'
  );
  assert.equal(
    await readFile(scoreboardMarkdown, 'utf8'),
    await readFile('docs/model-competition-scoreboard.md', 'utf8'),
    'competition scoreboard Markdown must ignore untracked benchmark receipts'
  );
} finally {
  await rm(untrackedReceiptPath, { force: true });
  await rm(outputDir, { recursive: true, force: true });
}

console.log('tracked-benchmark-evidence-contract.test: ok');
