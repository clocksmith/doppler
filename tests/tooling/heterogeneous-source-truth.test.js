import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createSourceTruthPacket, forgeModelIRV2 } from '../../src/converter/source-truth-forge.js';

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');

async function forgeCheckedInSpec(name) {
  const specPath = path.join(repoRoot, 'reports/model-ir-v2', `${name}.spec.json`);
  const receiptPath = path.join(repoRoot, 'reports/model-ir-v2', `${name}.model-ir-receipt.json`);
  const spec = JSON.parse(await fs.readFile(specPath, 'utf8'));
  const sources = {};
  for (const [artifactId, sourcePath] of Object.entries(spec.sources)) {
    sources[artifactId] = JSON.parse(await fs.readFile(path.join(repoRoot, sourcePath), 'utf8'));
  }
  delete spec.sources;
  const packet = createSourceTruthPacket(spec, sources);
  const receipt = forgeModelIRV2(packet, sources);
  const checkedIn = JSON.parse(await fs.readFile(receiptPath, 'utf8'));
  assert.deepEqual(receipt, checkedIn, `${name} receipt must be deterministically reproducible`);
  return { spec, sources, packet, receipt };
}

const qwen = await forgeCheckedInSpec('qwen3.8-27b');
const qwenIR = qwen.receipt.modelIR;
assert.deepEqual(qwenIR.components.map((component) => component.type), [
  'text-decoder', 'vision-encoder', 'projector', 'speculative-drafter',
]);
assert.equal(qwenIR.blockSchedules.find((schedule) => schedule.id === 'text.schedule').blocks.length, 64);
assert.equal(qwenIR.blockSchedules.find((schedule) => schedule.id === 'text.schedule').blocks[3].blockClassId, 'full-attention');
assert.deepEqual(qwenIR.stateSpaces.map((state) => state.kind), ['kv', 'recurrent', 'convolutional']);
assert.deepEqual(qwenIR.supportScope.loweredEntryPoints, ['text.generate']);
assert.deepEqual(qwenIR.supportScope.unloweredEntryPoints, ['vision.encode', 'speculative.generate']);
assert.ok(qwenIR.provenance.facts.every((fact) => (
  fact.disposition === 'accepted'
  && ['direct', 'derived'].includes(fact.confidence)
  && fact.validation.status === 'passed'
  && fact.evidence.length > 0
)));

const glimmer = await forgeCheckedInSpec('glimmer-30b');
const glimmerIR = glimmer.receipt.modelIR;
assert.deepEqual(glimmer.receipt.unresolvedFacts, []);
assert.equal(glimmerIR.blockSchedules.find((schedule) => schedule.id === 'text.schedule').blocks.length, 52);
assert.equal(glimmerIR.blockSchedules.find((schedule) => schedule.id === 'vision.schedule').blocks.length, 50);
assert.equal(glimmerIR.blockClasses.find((block) => block.id === 'sliding-attention').positional.theta, 500000);
const fullAttention = glimmerIR.blockClasses.find((block) => block.id === 'full-attention');
assert.equal(fullAttention.positional.theta, 0);
assert.equal(fullAttention.positional.type, 'no-rope');
assert.deepEqual(fullAttention.geometry.queryKeyNorm, { type: 'rmsnorm', withScale: false });
assert.equal(fullAttention.geometry.queryScale, 3.87);
assert.equal(fullAttention.geometry.outputGateType, 'sigmoid');
assert.equal(fullAttention.normalization.postNormPosition, 'sublayer-output-before-residual');
assert.equal(glimmerIR.outputHeads[0].properties.preSoftcapMultiplier, 0.19611613513818404);
assert.ok(glimmerIR.sourceIdentity.artifacts.some((artifact) => (
  artifact.role === 'pinned-reference-implementation-semantics'
)));
assert.deepEqual(glimmerIR.supportScope.loweredEntryPoints, []);
assert.deepEqual(glimmerIR.supportScope.unloweredEntryPoints, ['text.generate', 'vision.encode']);

const inferred = structuredClone(qwen.packet);
inferred.facts[0].confidence = 'family-inferred';
assert.throws(() => forgeModelIRV2(inferred, qwen.sources), /cannot enter signed ModelIR/);

const runtimeFiles = (await fs.readdir(path.join(repoRoot, 'src/client/runtime')))
  .filter((file) => file.endsWith('.js'));
for (const file of runtimeFiles) {
  const source = await fs.readFile(path.join(repoRoot, 'src/client/runtime', file), 'utf8');
  assert.doesNotMatch(source, /qwen|glimmer/i, `${file} must remain model-family agnostic`);
}

console.log('✔ heterogeneous-source-truth.test.js passed');
