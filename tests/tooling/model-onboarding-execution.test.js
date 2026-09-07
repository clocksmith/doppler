import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { tmpdir } from 'node:os';
import { runOnboardingExecution } from '../../src/tooling/model-onboarding-execution.js';
import { hashOnboardingFile } from '../../src/tooling/model-onboarding-custody.js';

// Synthetic stage ports test orchestration and custody, not model qualification.
const root = await fs.mkdtemp(path.join(tmpdir(), 'doppler-onboarding-execution-'));
const stages = ['conversion', 'source-reference', 'model-qualification', 'capsule-construction', 'capsule-qualification'];
const sourceIdentity = { repository: 'test/source', revision: 'a'.repeat(40) };
try {
  await fs.writeFile(path.join(root, 'input.json'), '{}');
  await fs.writeFile(path.join(root, 'weights.bin'), new Uint8Array([1, 2, 3]));
  await fs.writeFile(path.join(root, 'conversion.json'), '{}');
  const input = { path: 'input.json', digest: await hashOnboardingFile(path.join(root, 'input.json')) };
  const config = { schema: 'doppler.model-onboarding-execution/v1', operation: 'embed', stages,
    inputs: Object.fromEntries(['sourcePolicy', 'frozenReference', 'qualificationConfig', 'capsuleConfig', 'driver',
      'installedPackageReceipt', 'installedPackageArchive', 'license', 'application'].map(key => [key, input])),
    sourceFiles: [{ path: 'weights.bin', sizeBytes: 3, digest: await hashOnboardingFile(path.join(root, 'weights.bin')) }] };
  const calls = [];
  const checks = [];
  let fail = true;
  const options = { sourceRoot: root, outputDir: path.join(root, 'resumable'), sourceIdentity,
    conversionPath: path.join(root, 'conversion.json'), conversionDigest: await hashOnboardingFile(path.join(root, 'conversion.json')),
    async runStage(context) {
      calls.push(context.stage);
      await fs.writeFile(path.join(context.outputDir, 'payload.json'), JSON.stringify({ stage: context.stage }));
      if (context.stage === 'source-reference' && fail) throw new Error('Interrupted source comparison');
      const receipt = { schema: 'doppler.onboarding-stage-result/v1', stage: context.stage, passed: true,
        source: sourceIdentity, physicalExecution: ['model-qualification', 'capsule-qualification'].includes(context.stage), data: { payload: 'payload.json' } };
      await fs.writeFile(path.join(context.outputDir, 'result.json'), JSON.stringify(receipt));
      return 'result.json';
    },
    async verifyStage(context, receipt) {
      checks.push(context.stage);
      assert.equal(JSON.parse(await fs.readFile(path.join(context.outputDir, receipt.data.payload), 'utf8')).stage, context.stage);
      return true;
    } };
  await assert.rejects(runOnboardingExecution(config, options), /Interrupted source comparison/);
  const failure = await fs.readFile(path.join(options.outputDir, 'source-reference/attempt-0/failure.json'), 'utf8');
  fail = false;
  const result = await runOnboardingExecution(config, options);
  assert.equal(result.qualified, true); assert.equal(result.published, false);
  assert.deepEqual(calls, ['conversion', 'source-reference', ...stages.slice(1)]);
  assert.equal(await fs.readFile(path.join(options.outputDir, 'source-reference/attempt-0/failure.json'), 'utf8'), failure);
  const priorCalls = calls.length;
  const priorChecks = checks.length;
  assert.deepEqual(await runOnboardingExecution(config, options), result);
  assert.equal(calls.length, priorCalls, 'resume reuses complete immutable stages');
  assert.equal(checks.length, priorChecks + stages.length, 'resume revalidates stage evidence');
  await fs.writeFile(path.join(options.outputDir, 'conversion/attempt-0/payload.json'), 'corrupt');
  await assert.rejects(runOnboardingExecution(config, options), /Retained stage output changed/);
  assert.equal(calls.length, priorCalls, 'corruption cannot silently trigger replacement execution');
  await fs.writeFile(path.join(root, 'weights.bin'), new Uint8Array([0, 2, 3]));
  await assert.rejects(runOnboardingExecution(config, { ...options, outputDir: path.join(root, 'changed-source') }), /input changed/);
  await fs.writeFile(path.join(root, 'weights.bin'), new Uint8Array([1, 2, 3]));
  await assert.rejects(runOnboardingExecution({ ...config, stages: [...stages].reverse() }, options), /execution stages/);
  await assert.rejects(runOnboardingExecution(config, { ...options, outputDir: path.join(root, 'bad-evidence'), verifyStage: async () => false }), /evidence rejected/);
  const mutate = { ...options, outputDir: path.join(root, 'mutated-conversion'), async runStage(context) {
    const receipt = await options.runStage(context);
    await fs.writeFile(options.conversionPath, '{"changed":true}');
    return receipt;
  } };
  await assert.rejects(runOnboardingExecution(config, mutate), /conversion configuration changed/);
} finally { await fs.rm(root, { recursive: true, force: true }); }
console.log('model-onboarding-execution.test: passed (synthetic stage ports)');
