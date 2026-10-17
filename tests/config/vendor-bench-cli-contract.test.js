import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { assertCompareArtifactContracts, selectReleaseCompareResults } from '../../tools/vendor-bench.js';

function runVendorBench(args) {
  return spawnSync(process.execPath, ['tools/vendor-bench.js', ...args], {
    cwd: process.cwd(), encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
  });
}

{
  const result = runVendorBench(['list']);
  assert.equal(result.status, 0, result.stderr);
}

{
  const eligible = {
    dopplerModelId: 'test-model', compareLane: 'performance_comparable',
    fairness: { releaseClaimable: true }, pairedComparable: true,
  };
  const legacyLane = { ...eligible, compareLane: null };
  const blocked = [
    { ...eligible, fairness: undefined },
    { ...eligible, fairness: { claimGrade: true, localComparable: true, releaseClaimable: false } },
    { ...eligible, fairness: { releaseClaimable: 'true' } },
    { ...eligible, compareLane: 'diagnostic' },
    { ...eligible, pairedComparable: false },
    { ...eligible, dopplerModelId: 'uncovered-model' },
    { ...eligible, dopplerModelId: '' },
  ];
  const inputs = [blocked[0], eligible, ...blocked.slice(1), legacyLane];
  const before = structuredClone(inputs);
  assert.deepEqual(selectReleaseCompareResults(inputs, new Set(['test-model'])), [eligible, legacyLane]);
  assert.deepEqual(inputs, before, 'selection must not rewrite evidence');
  assert.deepEqual(selectReleaseCompareResults(inputs, new Set()), []);
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'vendor-contracts-'));
  try {
    const source = path.join(tempDir, 'policy.json');
    const content = '{"unit":"original"}\n';
    await fs.writeFile(source, content);
    const identity = { source, sourceSha256: createHash('sha256').update(content).digest('hex') };
    const report = {
      benchmarkPolicy: { ...identity }, compareConfig: { ...identity }, metricContract: { ...identity },
      harnesses: { doppler: { ...identity }, transformersjs: { ...identity } },
    };
    await assertCompareArtifactContracts(report, 'unit-receipt');
    for (const field of ['source', 'sourceSha256']) {
      const missing = structuredClone(report);
      delete missing.metricContract[field];
      await assert.rejects(assertCompareArtifactContracts(missing, 'unit-receipt'), new RegExp(`missing metricContract.${field}`));
    }
    const changed = path.join(tempDir, 'changed.json');
    await fs.writeFile(changed, '{"unit":"changed"}\n');
    const stale = structuredClone(report);
    stale.metricContract.source = changed;
    await assert.rejects(assertCompareArtifactContracts(stale, 'unit-receipt'), /stale metricContract hash/);
    stale.metricContract.source = path.join(tempDir, 'absent.json');
    await assert.rejects(assertCompareArtifactContracts(stale, 'unit-receipt'), /compare artifact source is missing/);
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

{
  const tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-vendor-bench-telemetry-'));
  try {
    const outputPath = path.join(tempDir, 'llamacpp-resource-telemetry.json');
    const payload = {
      schemaVersion: 1,
      kind: 'llamacpp-bench',
      metrics: {
        decodeTokensPerSec: 10,
        prefillTokensPerSec: 20,
        prefillMs: 100,
        decodeMs: 200,
        totalRunMs: 300,
        decodeMsPerTokenP50: 3,
        decodeMsPerTokenP95: 4,
        decodeMsPerTokenP99: 5,
      },
      environment: {
        host: {
          platform: process.platform,
          arch: process.arch,
          nodeVersion: process.version,
          osRelease: os.release(),
          cpuModel: os.cpus()[0]?.model ?? null,
        },
        browser: {
          userAgent: null,
          platform: null,
          language: null,
          vendor: null,
          executable: null,
          channel: null,
        },
        gpu: {
          api: 'vulkan',
          backend: 'vulkan',
          vendor: 'amd',
          architecture: null,
          device: 'unit',
          description: 'unit',
          hasF16: null,
          hasSubgroups: null,
          hasTimestampQuery: null,
        },
        runtime: {
          library: 'llama.cpp',
          version: 'unit',
          surface: 'native-vulkan',
          device: 'Vulkan',
          dtype: 'Q4_K_M',
          requestedDtype: null,
          executionProviderMode: 'vulkan',
          cacheMode: null,
          loadMode: 'local-gguf',
        },
      },
      metadata: {
        version: 'unit',
        model: 'unit-model',
      },
    };
    const script = `console.log(${JSON.stringify(JSON.stringify(payload))})`;
    const result = runVendorBench([
      'run',
      '--target', 'llamacpp-vulkan-gguf',
      '--workload', 'p064-d064-t0-k1',
      '--model', 'unit-model',
      '--resource-telemetry', 'on',
      '--resource-telemetry-interval-ms', '50',
      '--output', outputPath,
      '--',
      process.execPath,
      '--input-type=module',
      '-e',
      script,
    ]);
    assert.equal(result.status, 0, result.stderr);
    const record = JSON.parse(await fs.readFile(outputPath, 'utf8'));
    assert.equal(record.resourceTelemetry.schemaVersion, 1);
    assert.equal(record.resourceTelemetry.enabled, true);
    assert.equal(record.resourceTelemetry.sampling.intervalMs, 50);
    assert.equal(record.resourceTelemetry.label, 'llamacpp-vulkan-gguf');
    assert.equal(typeof record.resourceTelemetry.sampling.sampleCount, 'number');
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true });
  }
}

console.log('vendor-bench-cli-contract.test: ok');
