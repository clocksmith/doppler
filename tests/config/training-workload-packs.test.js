import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const tempDir = mkdtempSync(path.join(tmpdir(), 'doppler-training-workload-packs-'));
try {
  const outPath = path.join(tempDir, 'verify-output.json');
  const result = spawnSync(process.execPath, [
    'tools/verify-training-workload-packs.js',
    '--registry',
    'src/training/workload-packs/registry.json',
    '--out',
    outPath,
  ], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });

  assert.equal(result.status, 0, result.stderr);
  const payload = JSON.parse(readFileSync(outPath, 'utf8'));
  assert.equal(payload.ok, true);
  assert.equal(payload.workloadCount >= 4, true);
} finally {
  rmSync(tempDir, { recursive: true, force: true });
}

// stale registry detection: independent block using a single known-good workload
{
  const staleDir = mkdtempSync(path.join(tmpdir(), 'doppler-workload-stale-'));
  try {
    // workload files live in root/, registry files live in staleDir/ to avoid cross-contamination
    const { mkdirSync } = await import('node:fs');
    const rootDir = path.join(staleDir, 'root');
    mkdirSync(rootDir);

    const srcWorkload = readFileSync('src/training/workload-packs/lora-toy-tiny.json', 'utf8');
    writeFileSync(path.join(rootDir, 'lora-toy-tiny.json'), srcWorkload, 'utf8');

    // Generate a correct single-entry registry for that root
    const generatedRegistryPath = path.join(staleDir, 'registry.json');
    const writeResult = spawnSync(process.execPath, [
      'tools/verify-training-workload-packs.js',
      '--root', rootDir,
      '--registry', generatedRegistryPath,
      '--write-registry',
    ], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });
    assert.equal(writeResult.status, 0, writeResult.stderr);

    // Add a phantom entry to the generated registry
    const generated = JSON.parse(readFileSync(generatedRegistryPath, 'utf8'));
    const staleRegistry = {
      ...generated,
      workloads: [
        ...generated.workloads,
        {
          id: 'phantom-stale-workload',
          path: path.join(rootDir, 'phantom.json').replace(/\\/g, '/'),
          sha256: 'deadbeef000000000000000000000000',
          baselineReportId: 'trn_phantom-stale-workload_deadbeef0000',
          claimBoundary: 'phantom',
        },
      ],
    };
    const staleRegistryPath = path.join(staleDir, 'stale-registry.json');
    writeFileSync(staleRegistryPath, JSON.stringify(staleRegistry, null, 2), 'utf8');

    const staleResult = spawnSync(process.execPath, [
      'tools/verify-training-workload-packs.js',
      '--root', rootDir,
      '--registry', staleRegistryPath,
    ], {
      cwd: process.cwd(),
      encoding: 'utf8',
    });
    assert.notEqual(staleResult.status, 0, staleResult.stdout);
    assert.match(staleResult.stderr, /stale registry entries: phantom-stale-workload/);
  } finally {
    rmSync(staleDir, { recursive: true, force: true });
  }
}

console.log('training-workload-packs.test: ok');
