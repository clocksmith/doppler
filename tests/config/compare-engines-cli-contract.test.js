import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { spawnSync } from 'node:child_process';

function runCompareEngines(args) {
  return spawnSync(process.execPath, ['tools/compare-engines.js', ...args], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
}

{
  const result = runCompareEngines(['--help']);
  assert.equal(result.status, 0, result.stderr);
}

{
  const result = runCompareEngines([
    '--doppler-surface', 'invalid-surface',
    '--json',
  ]);
  assert.notEqual(result.status, 0);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-compare-config-'));
  const badConfigPath = path.join(tempDir, 'bad-compare-config.json');
  const badConfig = {
    schemaVersion: 1,
    updated: '2026-03-05',
    modelProfiles: [
      {
        dopplerModelId: 'gemma-3-270m-it-wf16-ef16-hf16',
        defaultTjsModelId: 'onnx-community/gemma-3-270m-it-ONNX',
        defaultKernelPath: null,
        modelBaseDir: 'local',
        defaultDopplerSurface: 'unsupported',
      },
    ],
  };
  await fs.writeFile(badConfigPath, `${JSON.stringify(badConfig, null, 2)}\n`, 'utf8');
  const result = runCompareEngines([
    '--compare-config', badConfigPath,
    '--json',
  ]);
  assert.notEqual(result.status, 0);
}

console.log('compare-engines-cli-contract.test: ok');
