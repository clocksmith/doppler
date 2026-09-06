import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { hashBytesSha256 } from '../../src/formats/canonical-hash.js';
import { runModelOnboarding } from '../../src/tooling/model-onboarding.js';
import { forgeSourceTruthFromFiles } from '../../src/tooling/source-truth-inputs.js';

const root = process.cwd();
const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'doppler-onboarding-'));
const tool = path.join(root, 'tools/forge-source-truth-model-ir-v2.js');
async function pin(file) {
  return { path: file, digest: hashBytesSha256(await fs.readFile(path.resolve(root, file))) };
}
async function json(file) { return JSON.parse(await fs.readFile(file, 'utf8')); }
async function writeInput(filename, value) {
  const file = path.join(temp, filename);
  await fs.writeFile(file, JSON.stringify(value));
  return pin(file);
}
const specPath = 'reports/model-ir-v2/qwen3.8-27b.spec.json';
const vocabularyPath = 'src/config/forge/lowering-vocabularies/hybrid-text-v1.json';
const recipePath = 'src/config/forge/lineage/qwen3.8-27b.json';
const recipe = await json(recipePath);
const config = {
  schema: 'doppler.model-onboarding/v1',
  sourceSpec: await pin(specPath), vocabulary: await pin(vocabularyPath),
  entryPointIds: ['text.generate'],
  lineage: { recipe: await pin(recipePath), template: await pin(recipe.template) },
};
const run = (value, directory) => runModelOnboarding(value, { sourceRoot: root, outputDir: path.join(temp, directory) });

try {
  // Real retained source metadata and Forge implementation; no weights or GPU execution.
  const result = await run(config, 'accepted');
  assert.equal(result.status, 'candidate-materialized');
  assert.equal(result.qualified, false);
  assert.equal(result.published, false);
  const output = path.join(temp, 'accepted');
  const lineage = await json(path.join(output, 'lineage-lowering-receipt.json'));
  assert.equal(lineage.conversionConfig.output.modelBaseId, recipe.modelId);
  assert.equal(lineage.rejectedCandidates.length, recipe.candidateAudit.rejected.length);
  assert.equal(lineage.conversionConfig.manifest.artifactIdentity.sourceRevision, result.sourceIdentity.revision);
  const assessment = await json(path.join(output, 'support-assessment.json'));
  assert.equal(assessment.tasks.length, 0);
  assert.equal(assessment.audits[0].lowerable, true);
  const originalStats = await fs.stat(path.join(output, 'conversion-config.json'));
  assert.deepEqual(await run(config, 'accepted'), result);
  assert.equal((await fs.stat(path.join(output, 'conversion-config.json'))).mtimeMs, originalStats.mtimeMs);
  await fs.unlink(path.join(output, 'onboarding-result.json'));
  assert.deepEqual(await run(config, 'accepted'), result, 'resume after retained stages without a final result');
  assert.equal((await fs.stat(path.join(output, 'conversion-config.json'))).mtimeMs, originalStats.mtimeMs);

  const noRecipe = await run({ ...config, lineage: null }, 'no-recipe');
  assert.equal(noRecipe.status, 'recipe-required');
  assert.equal(Object.hasOwn(noRecipe.outputs, 'conversion'), false);
  assert.equal(noRecipe.manualRequirements[0].kind, 'lineage-recipe');

  const glimmer = { ...config, sourceSpec: await pin('reports/model-ir-v2/glimmer-30b.spec.json'), lineage: null };
  const blocked = await run(glimmer, 'glimmer');
  assert.equal(blocked.status, 'blocked');
  assert.ok(blocked.manualRequirements.some((task) => task.kind === 'component-semantics'));
  assert.ok(blocked.manualRequirements.some((task) => task.kind === 'block-semantics' && task.sourceEvidence.length > 0));
  assert.match(JSON.stringify(blocked.manualRequirements), /embeddingNormalization|qkScale/);
  assert.equal(Object.hasOwn(blocked.outputs, 'conversion'), false);

  const vocabulary = await json(vocabularyPath);
  delete vocabulary.outputHeadContracts;
  const missingHead = await run({ ...config, vocabulary: await writeInput('missing-head.json', vocabulary) }, 'missing-head');
  assert.equal(missingHead.status, 'blocked');
  assert.ok(missingHead.manualRequirements.some((task) => task.kind === 'output-semantics'));

  for (const changes of [
    { schema: 'other' }, { customerId: 'not-required' }, { lineage: undefined },
    { entryPointIds: [] }, { entryPointIds: ['text.generate', 'text.generate'] },
    { entryPointIds: new Array(1) }, { entryPointIds: [' text.generate'] },
    { sourceSpec: { path: specPath, digest: [config.sourceSpec.digest] } },
    { lineage: { ...config.lineage, execute: true } },
  ]) await assert.rejects(run({ ...config, ...changes }, 'invalid'));

  const unpinnedSpec = await json(specPath);
  unpinnedSpec.sourceIdentity.revision = 'main';
  await assert.rejects(run({ ...config, sourceSpec: await writeInput('unpinned.json', unpinnedSpec) }, 'unpinned'), /immutable source revision/);
  const wrongHash = { ...config, vocabulary: { ...config.vocabulary, digest: `sha256:${'0'.repeat(64)}` } };
  await assert.rejects(run(wrongHash, 'changed-input'), /Onboarding input changed/);
  const failures = (await fs.readdir(path.join(temp, 'changed-input'))).filter((name) => name.startsWith('failure-'));
  assert.equal(failures.length, 1);
  assert.equal((await json(path.join(temp, 'changed-input', failures[0]))).stage, 'support-assessment');
  await assert.rejects(run(wrongHash, 'changed-input'), /Onboarding input changed/);
  assert.equal((await fs.readdir(path.join(temp, 'changed-input'))).filter((name) => name.startsWith('failure-')).length, 1);

  await fs.writeFile(path.join(output, 'conversion-config.json'), 'corrupt retained candidate');
  await assert.rejects(run(config, 'accepted'), /Retained onboarding output differs/);
  assert.equal(await fs.readFile(path.join(output, 'conversion-config.json'), 'utf8'), 'corrupt retained candidate');
  const rejectedRecipe = { ...recipe, compatibilityRequirements: [{ factId: 'full.headDim', equals: -1 }] };
  await assert.rejects(run({ ...config, lineage: { ...config.lineage, recipe: await writeInput('bad-recipe.json', rejectedRecipe) } }, 'bad-recipe'), /does not equal/);
  assert.equal((await fs.readdir(path.join(temp, 'bad-recipe'))).includes('conversion-config.json'), false);

  const configFile = (await writeInput('cli-config.json', config)).path;
  const cliResult = JSON.parse(execFileSync(process.execPath, [tool, '--config', configFile, '--out', path.join(temp, 'cli')], { cwd: temp, encoding: 'utf8' }));
  assert.equal(cliResult.status, 'candidate-materialized');
  const blockedFile = (await writeInput('cli-blocked.json', glimmer)).path;
  const cliBlocked = spawnSync(process.execPath, [tool, '--config', blockedFile, '--out', path.join(temp, 'cli-blocked')], { encoding: 'utf8' });
  assert.equal(cliBlocked.status, 1);
  assert.equal(JSON.parse(cliBlocked.stdout).status, 'blocked');
  const legacy = path.join(temp, 'legacy-receipt.json');
  execFileSync(process.execPath, [tool, '--spec', path.join(root, specPath), '--out', legacy], { cwd: temp });
  assert.deepEqual(await json(legacy), await json(path.join(output, 'model-ir-receipt.json')));
  assert.notEqual(spawnSync(process.execPath, [tool, '--spec', specPath, '--config', configFile, '--out', legacy]).status, 0);

  const spec = await json(specPath);
  const copy = structuredClone(spec);
  await forgeSourceTruthFromFiles(spec, root);
  assert.deepEqual(spec, copy, 'source reader may not mutate the input specification');
  const changedSource = await writeInput('changed-source.json', { ...await json(spec.sources.config), fabricated: true });
  spec.sources.config = changedSource.path;
  await assert.rejects(forgeSourceTruthFromFiles(spec, root), /hash mismatch/);

  const bytePinned = await json(specPath);
  const configSource = await pin(bytePinned.sources.config);
  bytePinned.sources.config = { path: configSource.path, hash: configSource.digest, format: 'json' };
  bytePinned.sourceIdentity.artifacts.find((artifact) => artifact.artifactId === 'config').hash = configSource.digest;
  assert.equal((await forgeSourceTruthFromFiles(bytePinned, root)).modelIR.modelId, bytePinned.modelId);
  bytePinned.sources.config.hash = `sha256:${'0'.repeat(64)}`;
  await assert.rejects(forgeSourceTruthFromFiles(bytePinned, root), /byte hash mismatch/);

  const malformedHeader = path.join(temp, 'malformed.safetensors');
  const prefix = Buffer.alloc(8);
  prefix.writeBigUInt64LE(2n ** 63n);
  await fs.writeFile(malformedHeader, prefix);
  bytePinned.sources = { headers: { path: malformedHeader, hash: `sha256:${'0'.repeat(64)}`, format: 'safetensors-header' } };
  await assert.rejects(forgeSourceTruthFromFiles(bytePinned, root), /header length.*unsupported/);
  await fs.writeFile(malformedHeader, Buffer.alloc(1));
  await assert.rejects(forgeSourceTruthFromFiles(bytePinned, root), /Truncated SafeTensors prefix/);
  prefix.writeBigUInt64LE(20n);
  await fs.writeFile(malformedHeader, prefix);
  await assert.rejects(forgeSourceTruthFromFiles(bytePinned, root), /Truncated SafeTensors header/);
} finally {
  await fs.rm(temp, { recursive: true, force: true });
}
console.log('model-onboarding.test: ok (source metadata to candidate; no physical qualification)');
