import assert from 'node:assert/strict';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { ToolingCommandError } from '../../src/tooling/command-envelope.js';
import { compareBenchResults, loadBaseline } from '../../src/cli/cli-output.js';
import { buildComparisonFingerprint } from '../../src/client/inspection.js';
import { getRuntimeConfig, setRuntimeConfig } from '../../src/config/runtime.js';

function result() {
  return {
    suite: 'bench', workload: 'inference', failed: 0, modelId: 'model',
    comparisonFingerprint: buildComparisonFingerprint({
      artifact: { modelId: 'model', manifestHash: 'a'.repeat(64) },
      tokenizer: { digest: 'tokenizer' }, promptTokenIds: [1, 2],
      sampling: { temperature: 0, topK: 1, topP: 1 }, observationPolicyId: 'demo/always-on',
      execution: { backend: 'webgpu', executionPlanId: 'plan' },
      adapter: { vendor: 'test', device: 'test' }, browser: { userAgent: 'test' },
    }),
    env: { library: 'doppler', runtime: 'browser', device: 'webgpu' },
    cacheMode: 'warm', loadMode: 'cached',
    metrics: { warmupRuns: 2, timedRuns: 3, seed: 42, maxTokens: 64, prompt: 'prompt', decodeTokensPerSec: 100 },
  };
}
const originalConfig = structuredClone(getRuntimeConfig());
try {
  const current = result();
  current.metrics.decodeTokensPerSec = 120;
  assert.deepEqual(compareBenchResults(current, result()).improvements, ['decode tok/s']);
  setRuntimeConfig({ shared: { benchmark: { comparison: { regressionThresholdPercent: 25 } } } });
  assert.deepEqual(compareBenchResults(current, result()).improvements, [], 'uses resolved policy threshold');
  for (const mutate of [
    (r) => { r.modelId = 'another-model'; },
    (r) => { delete r.comparisonFingerprint; },
    (r) => { r.comparisonFingerprint.identity.promptTokenIds = [99]; },
    (r) => { r.env.runtime = 'node'; },
    (r) => { r.env = {}; },
    (r) => { r.workload = 'embedding'; },
    (r) => { r.cacheMode = 'cold'; },
    (r) => { r.loadMode = 'network'; },
    (r) => { r.metrics.seed = 4; },
    (r) => { delete r.metrics.seed; },
    (r) => { r.metrics.timedRuns = 10; },
    (r) => { r.metrics.maxTokens = 128; },
    (r) => { r.metrics.prompt = 'another prompt'; },
    (r) => { r.failed = 1; },
  ]) {
    const invalid = result(); mutate(invalid);
    assert.throws(() => compareBenchResults(current, invalid), /comparison|fingerprint/i);
  }
  const anotherArtifact = result();
  anotherArtifact.comparisonFingerprint = buildComparisonFingerprint({
    artifact: { modelId: 'model', manifestHash: 'b'.repeat(64) },
    tokenizer: { digest: 'tokenizer' }, promptTokenIds: [1, 2], observationPolicyId: 'demo/always-on',
  });
  assert.throws(() => compareBenchResults(current, anotherArtifact), /fingerprints differ/);
  const directory = await mkdtemp(path.join(tmpdir(), 'doppler-baseline-'));
  try {
    for (const content of ['{invalid', 'null', '[]']) {
      await writeFile(path.join(directory, 'latest.json'), content);
      await assert.rejects(loadBaseline('last', directory), (error) =>
        error instanceof ToolingCommandError && error.code === 'benchmark_baseline_load_failed'
          && error.details.baselinePath === path.join(directory, 'latest.json'));
    }
    await assert.rejects(loadBaseline(path.join(directory, 'missing.json'), directory), (error) =>
      error instanceof ToolingCommandError && error.code === 'benchmark_baseline_load_failed');
    await writeFile(path.join(directory, 'latest.json'), JSON.stringify(result()));
    assert.deepEqual(await loadBaseline('last', directory), result());
  } finally { await rm(directory, { recursive: true, force: true }); }
} finally {
  setRuntimeConfig(originalConfig);
}
console.log('CLI benchmark comparison contract tests passed');
